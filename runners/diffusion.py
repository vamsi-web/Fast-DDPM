import os
import logging
import time
import glob

import numpy as np
import pandas as pd
import math
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry, calculate_psnr
from datasets import data_transform, inverse_data_transform
from datasets.pmub import PMUB
from datasets.LDFDCT import LDFDCT
from datasets.BRATS import BRATS
from functions.ckpt_util import get_ckpt_path
from skimage.metrics import structural_similarity as ssim
import torchvision.utils as tvu
import torchvision
from PIL import Image

class Diffusion:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = config.device
        self.num_timesteps = config.diffusion.num_diffusion_timesteps
        self.betas = torch.linspace(config.diffusion.beta_start, config.diffusion.beta_end, self.num_timesteps).to(self.device)

    def pet_train(self):
        """
        Training logic for the PET dataset.
        """
        args, config = self.args, self.config
        tb_logger = config.tb_logger

        # Initialize the PET dataset
        dataset = PETDataset(config.data.train_dataroot, config.data.image_size, split='train')
        print('Start training your Fast-DDPM model on PET dataset.')

        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=True
        )

        # Model initialization
        model = Model(config)
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        # Optimizer and EMA
        optimizer = get_optimizer(config, model.parameters())
        if config.model.ema:
            ema_helper = EMAHelper(mu=config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        # Resume training
        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if config.model.ema:
                ema_helper.load_state_dict(states[4])

        # Training loop
        for epoch in range(start_epoch, config.training.n_epochs):
            for i, batch in enumerate(train_loader):
                n = batch['LPET'].size(0)
                model.train()
                step += 1

                # Load LPET and FDPET
                x_lpet = batch['LPET'].to(self.device)
                x_fdpet = batch['FDPET'].to(self.device)

                e = torch.randn_like(x_fdpet)
                b = self.betas

                if self.args.scheduler_type == 'uniform':
                    skip = self.num_timesteps // self.args.timesteps
                    t_intervals = torch.arange(-1, self.num_timesteps, skip)
                    t_intervals[0] = 0
                elif self.args.scheduler_type == 'non-uniform':
                    t_intervals = torch.tensor([0, 199, 399, 599, 699, 799, 849, 899, 949, 999])
                else:
                    raise Exception("The scheduler type is either uniform or non-uniform.")
                    
                idx = torch.randint(0, len(t_intervals), size=(n,)).to(self.device)
                t = t_intervals[idx]

                # Compute loss
                loss = loss_registry[config.model.type](model, x_lpet, x_fdpet, t, e, b)

                tb_logger.add_scalar("loss", loss.item(), global_step=step)

                logging.info(f"Step: {step}, Loss: {loss.item()}")

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
                optimizer.step()

                # EMA update
                if config.model.ema:
                    ema_helper.update(model)

                # Save checkpoint
                if step % config.training.snapshot_freq == 0 or step == 1:
                    states = [model.state_dict(), optimizer.state_dict(), epoch, step]
                    if config.model.ema:
                        states.append(ema_helper.state_dict())
                    torch.save(states, os.path.join(self.args.log_path, f"ckpt_{step}.pth"))
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

            logging.info(f"Epoch {epoch+1}/{config.training.n_epochs} completed.")

        logging.info("Training finished.")

    def pet_sample(self):
        """
        Sampling logic for the PET dataset.
        """
        ckpt_list = self.config.sampling.ckpt_id
        for ckpt_idx in ckpt_list:
            self.ckpt_idx = ckpt_idx
            model = Model(self.config)
            print(f'Start inference on model checkpoint {ckpt_idx}.')

            # Load model checkpoint
            states = torch.load(
                os.path.join(self.args.log_path, f"ckpt_{ckpt_idx}.pth"),
                map_location=self.device,
            )
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)

            model.eval()

            # Example Sampling
            self.pet_generate_samples(model)

    def pet_generate_samples(self, model):
        """
        Generate PET samples from the model.
        """
        print("Generating samples for PET dataset...")
        sample_input = torch.randn((1, 1, 128, 128)).to(self.device)  # Example random input
        generated_samples = model(sample_input)
        print("Sample generation completed.")
        # Add visualization or save the generated samples
