import os, torch
import os.path as osp
import shutil
from tqdm.auto import tqdm
from einops import rearrange
from accelerate import Accelerator
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader, random_split, DistributedSampler
from catok.utils.logger import SmoothedValue, MetricLogger, empty_cache
from accelerate.utils import DistributedDataParallelKwargs
from torchmetrics.functional.image import (
    peak_signal_noise_ratio as psnr,
    structural_similarity_index_measure as ssim
)
from catok.engine.trainer_utils import (
    instantiate_from_config, concat_all_gather,
    save_img_batch, get_fid_stats,
    EMAModel, PaddedDataset, create_scheduler, load_state_dict,
    load_safetensors, setup_result_folders, create_optimizer
)

class MeanFlowTrainer:
    def __init__(
        self,
        model,
        dataset,
        test_dataset=None,
        test_only=False,
        num_epoch=400,
        valid_size=32,
        blr=1e-4,
        cosine_lr=True,
        lr_min=0,
        warmup_epochs=100,
        num_test_images=50000,
        warmup_steps=None,
        warmup_lr_init=0,
        decay_steps=None,
        batch_size=32,
        eval_bs=32,
        test_bs=64,
        num_workers=8,
        pin_memory=False,
        max_grad_norm=None,
        grad_accum_steps=1,
        precision='bf16',
        save_every=10000,
        sample_every=1000,
        fid_every=50000,
        result_folder=None,
        log_dir="./log",
        cfg=3.0,
        train_num_slots=None,
        test_num_slots=None,
        test_num_steps=1,
        eval_fid=False,
        fid_stats=None,
        fid_real_dir=None,
        enable_ema=False,
        compile=False,
    ):
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator = Accelerator(
            kwargs_handlers=[kwargs],
            mixed_precision=precision,
            gradient_accumulation_steps=grad_accum_steps,
            log_with="tensorboard",
            project_dir=log_dir,
        )

        self.model = instantiate_from_config(model)
        self.num_slots = model.params.num_slots
        self.train_num_slots = train_num_slots
        if self.train_num_slots is not None:
            self.train_num_slots = min(self.train_num_slots, self.num_slots)

        self.fid_real_dir = fid_real_dir
        if test_dataset is not None:
            test_dataset = instantiate_from_config(test_dataset)
            self.test_ds = test_dataset
            
            # Calculate padded dataset size to ensure even distribution
            total_size = len(test_dataset)
            world_size = self.accelerator.num_processes
            padding_size = world_size * test_bs - (total_size % (world_size * test_bs))
            self.test_dataset_size = total_size
            
            self.test_ds = PaddedDataset(self.test_ds, padding_size)
            self.test_dl = DataLoader(
                self.test_ds,
                batch_size=test_bs,
                num_workers=num_workers,
                pin_memory=pin_memory,
                shuffle=False,
                drop_last=True,
            )
            if self.accelerator.is_main_process:
                print(f"test dataset size: {len(test_dataset)}, test batch size: {test_bs}")
            if self.fid_real_dir is None:
                self.fid_real_dir = getattr(self.test_ds, "fid_real_dir", None)
        else:
            self.test_dl = None
        self.test_only = test_only

        if not test_only:
            dataset = instantiate_from_config(dataset)
            train_size = len(dataset) - valid_size
            self.train_ds, self.valid_ds = random_split(
                dataset,
                [train_size, valid_size],
                generator=torch.Generator().manual_seed(42),
            )
            if self.accelerator.is_main_process:
                print(f"train dataset size: {train_size}, valid dataset size: {valid_size}")

            sampler = DistributedSampler(
                self.train_ds,
                num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
                shuffle=True,
            )
            self.train_dl = DataLoader(
                self.train_ds,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=True,
            )
            self.valid_dl = DataLoader(
                self.valid_ds,
                batch_size=eval_bs,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
            )

            effective_bs = batch_size * grad_accum_steps * self.accelerator.num_processes
            lr = blr * effective_bs / 256
            if self.accelerator.is_main_process:
                print(f"Effective batch size is {effective_bs}")

            self.g_optim = create_optimizer(self.model, weight_decay=0.0, learning_rate=lr,) # accelerator=self.accelerator)
            
            if warmup_epochs is not None:
                warmup_steps = warmup_epochs * len(self.train_dl)
            
            self.g_sched = create_scheduler(
                self.g_optim,
                num_epoch,
                len(self.train_dl),
                lr_min,
                warmup_steps,
                warmup_lr_init,
                decay_steps,
                cosine_lr
            )
            self.accelerator.register_for_checkpointing(self.g_sched)
            if test_dataset is not None:
                self.model, self.g_optim, self.g_sched, self.test_dl = self.accelerator.prepare(self.model, self.g_optim, self.g_sched, self.test_dl)
            else:
                self.model, self.g_optim, self.g_sched = self.accelerator.prepare(self.model, self.g_optim, self.g_sched)
        else:
            self.model, self.test_dl = self.accelerator.prepare(self.model, self.test_dl)

        self.steps = 0
        self.loaded_steps = -1

        if compile:
            _model = self.accelerator.unwrap_model(self.model)
            _model.vae = torch.compile(_model.vae, mode="reduce-overhead")
            _model.dit = torch.compile(_model.dit, mode="reduce-overhead")
            _model.encoder = torch.compile(_model.encoder, mode="reduce-overhead") # nan loss when compiled together with dit, no idea why
            _model.encoder2slot = torch.compile(_model.encoder2slot, mode="reduce-overhead")

        self.enable_ema = enable_ema
        if self.enable_ema and not self.test_only: # when testing, we directly load the ema dict and skip here
            self.ema_model = EMAModel(self.accelerator.unwrap_model(self.model), self.device)
            self.accelerator.register_for_checkpointing(self.ema_model)

        self._load_checkpoint(model.params.ckpt_path)
        if self.test_only:
            self.steps = self.loaded_steps

        self.num_epoch = num_epoch
        self.save_every = save_every
        self.sample_every = sample_every
        self.fid_every = fid_every
        self.max_grad_norm = max_grad_norm

        self.cfg = cfg
        self.test_num_slots = test_num_slots
        self.test_num_steps = test_num_steps
        if self.test_num_slots is not None:
            self.test_num_slots = min(self.test_num_slots, self.num_slots)
        else:
            self.test_num_slots = self.num_slots
        eval_fid = eval_fid or model.params.eval_fid # legacy
        self.eval_fid = eval_fid
        if eval_fid:
            assert test_dataset is not None
        self.fid_stats = fid_stats
        if self.eval_fid and self.fid_real_dir is None:
            self.fid_real_dir = "./dataset/imagenet/val256"

        self.result_folder = result_folder
        self.model_saved_dir, self.image_saved_dir = setup_result_folders(result_folder)

    @property
    def device(self):
        return self.accelerator.device

    def _load_checkpoint(self, ckpt_path=None):
        if ckpt_path is None or not osp.exists(ckpt_path):
            return
        
        model = self.accelerator.unwrap_model(self.model)

        if osp.isdir(ckpt_path):
            # ckpt_path is something like 'path/to/models/step10/'
            self.loaded_steps = int(
                ckpt_path.split("step")[-1].split("/")[0]
            )
            if not self.test_only:
                self.accelerator.load_state(ckpt_path, strict=False)
            else:
                model_path = osp.join(ckpt_path, "model.safetensors")
                if osp.exists(model_path):
                    load_safetensors(model_path, model)
                    if self.accelerator.is_main_process:
                        print(f"Loaded full model from {model_path}")
                if self.enable_ema:
                    model_path = osp.join(ckpt_path, "custom_checkpoint_1.pkl")
                    if osp.exists(model_path):
                        state_dict = torch.load(model_path, map_location="cpu")
                        load_state_dict(state_dict, model)
                        if self.accelerator.is_main_process:
                            print(f"Loaded ema model from {model_path}")
        else:
            # ckpt_path is something like 'path/to/models/step10.pt'
            if ckpt_path.endswith(".safetensors"):
                load_safetensors(ckpt_path, model)
            else:
                state_dict = torch.load(ckpt_path, map_location="cpu")
                load_state_dict(state_dict, model)
        if self.accelerator.is_main_process:
            print(f"Loaded checkpoint from {ckpt_path}")

    def train(self, config=None):
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.accelerator.is_main_process:
            print(f"number of learnable parameters: {n_parameters//1e6}M")
        if config is not None:
            # save the config
            from omegaconf import OmegaConf
            if isinstance(config, str) and osp.exists(config):
                # If it's a path, copy the file to config.yaml
                shutil.copy(config, osp.join(self.result_folder, "config.yaml"))
            else:
                # If it's an OmegaConf object, dump it
                config_save_path = osp.join(self.result_folder, "config.yaml")
                OmegaConf.save(config, config_save_path)

        self.accelerator.init_trackers("catok")

        if self.test_only:
            empty_cache()
            self.evaluate()
            self.accelerator.wait_for_everyone()
            empty_cache()
            return

        for epoch in range(self.num_epoch):
            if ((epoch + 1) * len(self.train_dl)) <= self.loaded_steps:
                if self.accelerator.is_main_process:
                    print(f"Epoch {epoch} is skipped because it is loaded from ckpt")
                self.steps += len(self.train_dl)
                continue

            if self.steps < self.loaded_steps:
                for _ in self.train_dl:
                    self.steps += 1
                    if self.steps >= self.loaded_steps:
                        break
            
            
            self.accelerator.unwrap_model(self.model).current_epoch = epoch
            self.model.train()  # Set model to training mode

            logger = MetricLogger(delimiter="  ")
            logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
            header = 'Epoch: [{}/{}]'.format(epoch, self.num_epoch)
            print_freq = 20
            for data_iter_step, batch in enumerate(logger.log_every(self.train_dl, print_freq, header)):
                img, _ = batch
                img = img.to(self.device, non_blocking=True)
                self.steps += 1

                with self.accelerator.accumulate(self.model):
                    with self.accelerator.autocast():
                        if self.steps == 1:
                            print(f"Training batch size: {img.size(0)}")
                            print(f"Hello from index {self.accelerator.local_process_index}")
                        if self.train_num_slots is None:
                            losses = self.model(img, epoch=epoch)
                        else:
                            losses = self.model(
                                img,
                                epoch=epoch,
                                inference_with_n_slots=self.train_num_slots,
                            )
                        # combine
                        loss = sum([v for _, v in losses.items()])

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients and self.max_grad_norm is not None:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.g_optim.step()
                    if self.g_sched is not None:
                        self.g_sched.step_update(self.steps)
                    self.g_optim.zero_grad()

                self.accelerator.wait_for_everyone()

                # update ema with state dict
                if self.enable_ema:
                    self.ema_model.update(self.accelerator.unwrap_model(self.model))

                for key, value in losses.items():
                    logger.update(**{key: value.item()})
                logger.update(lr=self.g_optim.param_groups[0]["lr"])

                if self.steps % self.save_every == 0:
                    self.save()

                if (self.steps % self.sample_every == 0) or (self.steps % self.fid_every == 0):
                    empty_cache()
                    self.evaluate()
                    self.accelerator.wait_for_everyone()
                    empty_cache()

                write_dict = dict(epoch=epoch)
                for key, value in losses.items(): # omitted all_gather here
                    write_dict.update(**{key: value.item()})
                write_dict.update(lr=self.g_optim.param_groups[0]["lr"])
                self.accelerator.log(write_dict, step=self.steps)

            logger.synchronize_between_processes()
            if self.accelerator.is_main_process:
                print("Averaged stats:", logger)

        self.save()
        self.accelerator.end_training()
        if self.accelerator.is_main_process:
            print("Train finished!")

    def save(self):
        self.accelerator.wait_for_everyone()
        save_dir = os.path.join(self.model_saved_dir, f"step{self.steps}")
        os.makedirs(save_dir, exist_ok=True)
        scaler = self.accelerator.scaler
        if not self.accelerator.is_main_process:
            self.accelerator.scaler = None
        self.accelerator.save_state(save_dir)
        if not self.accelerator.is_main_process:
            self.accelerator.scaler = scaler
        self.accelerator.wait_for_everyone()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        if not self.test_only:
            with tqdm(
                self.valid_dl,
                dynamic_ncols=True,
                disable=not self.accelerator.is_main_process,
            ) as valid_dl:
                for batch_i, batch in enumerate(valid_dl):
                    if isinstance(batch, tuple) or isinstance(batch, list):
                        img, targets = batch[0], batch[1]
                    else:
                        img = batch

                    img = img.to(self.device, non_blocking=True)
                    with self.accelerator.autocast():
                        rec = self.model(img, sample=True, inference_with_n_slots=self.test_num_slots, cfg=1.0, sample_steps=self.test_num_steps)
                    imgs_and_recs = torch.stack((img.to(rec.device), rec), dim=0)
                    imgs_and_recs = rearrange(imgs_and_recs, "r b ... -> (b r) ...")
                    imgs_and_recs = imgs_and_recs.detach().cpu().float()

                    grid = make_grid(
                        imgs_and_recs, nrow=6, normalize=True, value_range=(0, 1)
                    )
                    if self.accelerator.is_main_process:
                        save_image(
                            grid,
                            os.path.join(
                                self.image_saved_dir, f"step_{self.steps}_slots{self.test_num_slots}_{batch_i}_step{self.test_num_steps}.jpg"
                            ),
                        )

                    if self.cfg != 1.0:
                        with self.accelerator.autocast():
                            rec = self.model(img, sample=True, inference_with_n_slots=self.test_num_slots, cfg=self.cfg, sample_steps=self.test_num_steps)

                        imgs_and_recs = torch.stack((img.to(rec.device), rec), dim=0)
                        imgs_and_recs = rearrange(imgs_and_recs, "r b ... -> (b r) ...")
                        imgs_and_recs = imgs_and_recs.detach().cpu().float()

                        grid = make_grid(
                            imgs_and_recs, nrow=6, normalize=True, value_range=(0, 1)
                        )
                        if self.accelerator.is_main_process:
                            save_image(
                                grid,
                                os.path.join(
                                    self.image_saved_dir, f"step_{self.steps}_cfg_{self.cfg}_slots{self.test_num_slots}_step{self.test_num_steps}_{batch_i}.jpg"
                                ),
                            )
        if (self.eval_fid and self.test_dl is not None) and (self.test_only or (self.steps % self.fid_every == 0)):
            real_dir = self.fid_real_dir
            rec_dir = os.path.join(self.image_saved_dir, f"rec_step{self.steps}_slots{self.test_num_slots}_steps{self.test_num_steps}")
            os.makedirs(rec_dir, exist_ok=True)
            
            if self.cfg != 1.0:
                rec_cfg_dir = os.path.join(self.image_saved_dir, f"rec_step{self.steps}_cfg_{self.cfg}_slots{self.test_num_slots}_step{self.test_num_steps}")
                os.makedirs(rec_cfg_dir, exist_ok=True)

            def process_batch(cfg_value, save_dir, header):
                logger = MetricLogger(delimiter="  ")
                print_freq = 5
                psnr_values = []
                ssim_values = []
                total_processed = 0
                
                for batch_i, batch in enumerate(logger.log_every(self.test_dl, print_freq, header)):
                    imgs, targets = (batch[0], batch[1]) if isinstance(batch, (tuple, list)) else (batch, None)
                    
                    # Skip processing if we've already processed all real samples
                    if total_processed >= self.test_dataset_size:
                        break
                        
                    imgs = imgs.to(self.device, non_blocking=True)
                    if targets is not None:
                        targets = targets.to(self.device, non_blocking=True)

                    with self.accelerator.autocast():
                        recs = self.model(imgs, sample=True, inference_with_n_slots=self.test_num_slots, cfg=cfg_value, sample_steps=self.test_num_steps)

                    psnr_val = psnr(recs, imgs, data_range=1.0)
                    ssim_val = ssim(recs, imgs, data_range=1.0)
                    
                    recs = concat_all_gather(recs).detach()
                    psnr_val = concat_all_gather(psnr_val.view(1))
                    ssim_val = concat_all_gather(ssim_val.view(1))

                    # Remove padding after gathering from all GPUs
                    samples_in_batch = min(
                        recs.size(0),  # Always use the gathered size
                        self.test_dataset_size - total_processed
                    )
                    recs = recs[:samples_in_batch]
                    psnr_val = psnr_val[:samples_in_batch]
                    ssim_val = ssim_val[:samples_in_batch]
                    psnr_values.append(psnr_val)
                    ssim_values.append(ssim_val)

                    if self.accelerator.is_main_process:
                        rec_paths = [os.path.join(save_dir, f"step_{self.steps}_slots{self.test_num_slots}_{batch_i}_{j}_rec_cfg_{cfg_value}_steps{self.test_num_steps}.png") 
                                   for j in range(recs.size(0))]
                        save_img_batch(recs.cpu(), rec_paths)
                    
                    total_processed += samples_in_batch

                    self.accelerator.wait_for_everyone()
                    
                return torch.cat(psnr_values).mean(), torch.cat(ssim_values).mean()

            # Helper function to calculate and log metrics
            def calculate_and_log_metrics(real_dir, rec_dir, cfg_value, psnr_val, ssim_val):
                if self.accelerator.is_main_process:
                    print(f"{'CFG: {cfg_value}'} "
                          f"PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")
                    metrics_dict = get_fid_stats(real_dir, rec_dir, self.fid_stats)
                    fid = metrics_dict["frechet_inception_distance"]
                    inception_score = metrics_dict["inception_score_mean"]
                    
                    metric_prefix = "fid"
                    isc_prefix = "isc"
                    self.accelerator.log({
                        metric_prefix: fid,
                        isc_prefix: inception_score,
                        f"psnr": psnr_val,
                        f"ssim": ssim_val,
                        "cfg": cfg_value
                    }, step=self.steps)
                    
                    print(f"{'CFG: {cfg_value}'} "
                          f"FID: {fid:.2f}, ISC: {inception_score:.2f}, "
                          f"PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")

            # Process without CFG
            if self.cfg == 1.0 or not self.test_only:
                psnr_val, ssim_val = process_batch(1.0, rec_dir, 'Testing: w/o CFG')
                calculate_and_log_metrics(real_dir, rec_dir, 1.0, psnr_val, ssim_val)

            # Process with CFG if needed
            if self.cfg != 1.0:
                psnr_val, ssim_val = process_batch(self.cfg, rec_cfg_dir, 'Testing: w/ CFG')
                calculate_and_log_metrics(real_dir, rec_cfg_dir, self.cfg, psnr_val, ssim_val)

            # Cleanup
            if self.accelerator.is_main_process:
                shutil.rmtree(rec_dir)
                if self.cfg != 1.0:
                    shutil.rmtree(rec_cfg_dir)
        self.model.train()
