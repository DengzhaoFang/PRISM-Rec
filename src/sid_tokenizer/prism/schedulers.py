"""
Learning Rate Schedulers for Training

Implements various LR scheduling strategies including warmup + cosine annealing.
"""

import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class WarmupCosineScheduler(LambdaLR):
    """
    Warmup + Cosine Annealing LR Scheduler with optional minimum LR.
    
    Learning rate schedule:
    1. Linear warmup from 0 to initial LR (warmup_steps)
    2. Cosine annealing from initial LR to min_lr (remaining steps)
    
    Following best practices from TIGER and modern transformer training.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        num_cycles: float = 0.5,
        last_epoch: int = -1
    ):
        """
        Initialize scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            min_lr_ratio: Minimum LR as ratio of initial LR (default: 0.1)
            num_cycles: Number of cosine cycles (default: 0.5 for half-cosine)
            last_epoch: Last epoch index (for resuming)
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.num_cycles = num_cycles
        
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
    
    def lr_lambda(self, step: int) -> float:
        """
        Compute LR multiplier for given step.
        
        Args:
            step: Current training step
            
        Returns:
            LR multiplier (multiply with base LR to get actual LR)
        """
        if step < self.warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, self.warmup_steps))
        
        if step <= self.total_steps:
            # Cosine annealing
            progress = float(step - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            cosine_factor = 0.5 * (
                1.0 + math.cos(math.pi * self.num_cycles * 2.0 * progress)
            )
            return max(
                self.min_lr_ratio,
                self.min_lr_ratio + cosine_factor * (1.0 - self.min_lr_ratio)
            )
        else:
            # After total_steps, use minimum LR
            return self.min_lr_ratio


class InverseSquareRootScheduler:
    """
    Inverse square root LR scheduler with warmup.
    
    Used in Transformer and TIGER papers:
    - Constant LR during warmup
    - Inverse sqrt decay after warmup
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        init_lr: float,
        min_lr: float = 1e-6
    ):
        """
        Initialize scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps (constant LR)
            init_lr: Initial learning rate
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.current_step = 0
    
    def step(self):
        """Update learning rate"""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Constant during warmup
            lr = self.init_lr
        else:
            # Inverse square root decay
            decay_factor = math.sqrt(self.warmup_steps / self.current_step)
            lr = max(self.init_lr * decay_factor, self.min_lr)
        
        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def state_dict(self):
        """Return scheduler state"""
        return {
            'current_step': self.current_step,
            'warmup_steps': self.warmup_steps,
            'init_lr': self.init_lr,
            'min_lr': self.min_lr
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state"""
        self.current_step = state_dict['current_step']
        self.warmup_steps = state_dict['warmup_steps']
        self.init_lr = state_dict['init_lr']
        self.min_lr = state_dict['min_lr']


class ExponentialSchedulerWithWarmup:
    """
    Exponential decay with linear warmup.
    
    Simple but effective for many tasks.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        decay_rate: float = 0.96,
        decay_steps: int = 1000,
        min_lr: float = 1e-6
    ):
        """
        Initialize scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            decay_rate: Exponential decay rate
            decay_steps: Decay every N steps
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        self.current_step = 0
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self):
        """Update learning rate"""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Exponential decay
            decay_steps_elapsed = (self.current_step - self.warmup_steps) // self.decay_steps
            lr = self.base_lr * (self.decay_rate ** decay_steps_elapsed)
            lr = max(lr, self.min_lr)
        
        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']

