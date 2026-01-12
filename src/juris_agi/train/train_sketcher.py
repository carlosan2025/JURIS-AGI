#!/usr/bin/env python3
"""
Training script for the Neural Sketcher model.

This script trains the sketcher to predict primitive sequences from ARC-like tasks.
Training data is generated synthetically by:
1. Randomly composing primitives to create programs
2. Generating random input grids
3. Applying the programs to get output grids
4. Using (input, output, program) as training examples

Usage:
    python -m juris_agi.train.train_sketcher --epochs 100 --output models/sketcher.pt
"""

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import json

import numpy as np

from ..core.types import Grid, ARCTask, ARCPair
from ..dsl.ast import ASTNode, PrimitiveNode, ComposeNode, LiteralNode
from ..dsl.primitives import PRIMITIVES, list_primitives
from ..dsl.interpreter import DSLInterpreter
from ..cre.sketcher_model import (
    TORCH_AVAILABLE,
    SketcherConfig,
    PRIMITIVE_TO_ID,
    SOS_ID,
    EOS_ID,
    PAD_ID,
    NUM_PRIMITIVES,
)

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from ..cre.sketcher_model import SketcherTransformer


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SketcherTrainingConfig:
    """Configuration for sketcher training."""
    # Model config
    model_config: SketcherConfig = field(default_factory=SketcherConfig)

    # Training hyperparameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 100
    gradient_clip: float = 1.0

    # Data generation
    num_train_samples: int = 10000
    num_val_samples: int = 1000
    max_program_length: int = 4
    min_grid_size: int = 3
    max_grid_size: int = 10
    num_train_pairs: int = 3

    # Checkpointing
    checkpoint_every: int = 10
    output_dir: str = "models"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_program_length": self.max_program_length,
            "num_train_samples": self.num_train_samples,
        }


# =============================================================================
# Synthetic Data Generation
# =============================================================================

# Simple primitives that don't require special arguments
SIMPLE_PRIMITIVES = [
    "identity",
    "reflect_h",
    "reflect_v",
    "transpose",
    "crop_to_content",
]

# Primitives with int argument
INT_ARG_PRIMITIVES = {
    "rotate90": [1, 2, 3],
    "scale": [2],
    "tile_h": [2, 3],
    "tile_v": [2, 3],
}

# Primitives with color arguments
COLOR_ARG_PRIMITIVES = {
    "recolor": lambda: (random.randint(1, 9), random.randint(1, 9)),
    "fill": lambda: (random.randint(1, 9),),
}


def generate_random_grid(
    min_size: int = 3,
    max_size: int = 10,
    num_colors: int = 4,
) -> Grid:
    """Generate a random ARC-like grid."""
    height = random.randint(min_size, max_size)
    width = random.randint(min_size, max_size)

    # Use a small palette
    colors = [0] + random.sample(range(1, 10), min(num_colors - 1, 9))

    # Create grid with random pattern
    data = np.zeros((height, width), dtype=np.int32)

    # Fill with random pattern
    pattern_type = random.choice(["random", "structured", "sparse"])

    if pattern_type == "random":
        for r in range(height):
            for c in range(width):
                data[r, c] = random.choice(colors)

    elif pattern_type == "structured":
        # Create a simple structured pattern
        block_h = random.randint(1, max(1, height // 2))
        block_w = random.randint(1, max(1, width // 2))

        for r in range(height):
            for c in range(width):
                block_id = (r // block_h + c // block_w) % len(colors)
                data[r, c] = colors[block_id]

    else:  # sparse
        # Mostly background with some colored pixels
        num_colored = random.randint(1, (height * width) // 3)
        for _ in range(num_colored):
            r = random.randint(0, height - 1)
            c = random.randint(0, width - 1)
            data[r, c] = random.choice(colors[1:]) if len(colors) > 1 else 1

    return Grid(data)


def generate_random_program(max_length: int = 4) -> Tuple[ASTNode, List[str]]:
    """
    Generate a random program by composing primitives.

    Returns:
        Tuple of (AST, list of primitive names)
    """
    length = random.randint(1, max_length)
    primitives = []
    nodes = []

    for _ in range(length):
        # Choose primitive type
        prim_type = random.choice(["simple", "int_arg", "color_arg"])

        if prim_type == "simple":
            name = random.choice(SIMPLE_PRIMITIVES)
            primitives.append(name)
            nodes.append(PrimitiveNode(name))

        elif prim_type == "int_arg":
            name = random.choice(list(INT_ARG_PRIMITIVES.keys()))
            arg = random.choice(INT_ARG_PRIMITIVES[name])
            primitives.append(name)
            nodes.append(PrimitiveNode(name, [LiteralNode(arg)]))

        else:  # color_arg
            name = random.choice(list(COLOR_ARG_PRIMITIVES.keys()))
            args = COLOR_ARG_PRIMITIVES[name]()
            primitives.append(name)
            nodes.append(PrimitiveNode(name, [LiteralNode(a) for a in args]))

    if len(nodes) == 1:
        return nodes[0], primitives
    return ComposeNode(nodes), primitives


def generate_synthetic_task(
    config: Optional[SketcherTrainingConfig] = None,
) -> Tuple[ARCTask, ASTNode, List[str]]:
    """
    Generate a synthetic training task.

    Returns:
        Tuple of (ARCTask, program AST, primitive names)
    """
    config = config or SketcherTrainingConfig()

    # Generate a random program
    program, primitives = generate_random_program(config.max_program_length)

    # Create interpreter
    interpreter = DSLInterpreter()

    # Generate training pairs
    train_pairs = []
    for _ in range(config.num_train_pairs):
        # Generate input grid
        input_grid = generate_random_grid(
            config.min_grid_size,
            config.max_grid_size,
        )

        # Apply program to get output
        try:
            output_grid = interpreter.eval(program, {"grid": input_grid})
            if isinstance(output_grid, Grid):
                train_pairs.append(ARCPair(input=input_grid, output=output_grid))
        except Exception:
            # If program fails on this input, try again with identity
            train_pairs.append(ARCPair(input=input_grid, output=input_grid))

    if not train_pairs:
        # Fallback: identity task
        input_grid = generate_random_grid(config.min_grid_size, config.max_grid_size)
        train_pairs.append(ARCPair(input=input_grid, output=input_grid))
        program = PrimitiveNode("identity")
        primitives = ["identity"]

    task = ARCTask(
        task_id=f"synthetic_{random.randint(0, 999999):06d}",
        train=train_pairs,
        test=[],
    )

    return task, program, primitives


# =============================================================================
# PyTorch Dataset
# =============================================================================

if TORCH_AVAILABLE:

    class SyntheticSketcherDataset(Dataset):
        """Dataset of synthetic (task, program) pairs."""

        def __init__(
            self,
            num_samples: int,
            config: SketcherTrainingConfig,
        ):
            self.config = config
            self.samples = []

            print(f"Generating {num_samples} synthetic training samples...")
            for i in range(num_samples):
                task, program, primitives = generate_synthetic_task(config)

                # Convert primitives to token IDs
                token_ids = [SOS_ID]
                for prim in primitives:
                    if prim in PRIMITIVE_TO_ID:
                        token_ids.append(PRIMITIVE_TO_ID[prim])
                token_ids.append(EOS_ID)

                self.samples.append({
                    "task": task,
                    "program": program,
                    "primitives": primitives,
                    "token_ids": token_ids,
                })

                if (i + 1) % 1000 == 0:
                    print(f"  Generated {i + 1}/{num_samples} samples")

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            return self.samples[idx]


    def collate_fn(
        batch: List[Dict[str, Any]],
        max_seq_len: int = 8,
    ) -> Dict[str, Any]:
        """Collate function for DataLoader."""
        # Get max sequence length in batch
        max_len = min(max(len(s["token_ids"]) for s in batch), max_seq_len + 2)

        # Pad sequences
        padded_seqs = []
        for sample in batch:
            seq = sample["token_ids"][:max_len]
            padded = seq + [PAD_ID] * (max_len - len(seq))
            padded_seqs.append(padded)

        # Convert grids to tensors
        input_grids = []
        output_grids = []

        for sample in batch:
            task = sample["task"]
            for pair in task.train:
                input_grids.append(torch.tensor(pair.input.data, dtype=torch.long))
                output_grids.append(torch.tensor(pair.output.data, dtype=torch.long))

        return {
            "sequences": torch.tensor(padded_seqs, dtype=torch.long),
            "input_grids": input_grids,
            "output_grids": output_grids,
            "tasks": [s["task"] for s in batch],
        }


# =============================================================================
# Trainer
# =============================================================================

class SketcherTrainer:
    """Trainer for the neural sketcher model."""

    def __init__(self, config: SketcherTrainingConfig):
        self.config = config

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for training")

        # Create model
        self.model = SketcherTransformer(config.model_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

    def train_epoch(self, dataloader: "DataLoader") -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            self.optimizer.zero_grad()

            # Move data to device
            sequences = batch["sequences"].to(self.device)
            input_grids = [g.unsqueeze(0).to(self.device) for g in batch["input_grids"]]
            output_grids = [g.unsqueeze(0).to(self.device) for g in batch["output_grids"]]

            # Forward pass (teacher forcing)
            input_seq = sequences[:, :-1]  # All but last token
            target_seq = sequences[:, 1:]  # All but first token

            # For simplicity, use first training pair only
            # In practice, would handle variable number of pairs
            batch_size = sequences.shape[0]
            num_pairs = len(input_grids) // batch_size

            batch_inputs = []
            batch_outputs = []
            for i in range(min(num_pairs, 1)):  # Use first pair
                for j in range(batch_size):
                    idx = j * num_pairs + i
                    if idx < len(input_grids):
                        batch_inputs.append(input_grids[idx])
                        batch_outputs.append(output_grids[idx])

            if batch_inputs:
                logits = self.model(batch_inputs, batch_outputs, input_seq)

                # Compute loss
                loss = self.criterion(
                    logits.reshape(-1, logits.shape[-1]),
                    target_seq.reshape(-1),
                )

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )

                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                self.global_step += 1

        return total_loss / max(1, num_batches)

    def validate(self, dataloader: "DataLoader") -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                sequences = batch["sequences"].to(self.device)
                input_grids = [g.unsqueeze(0).to(self.device) for g in batch["input_grids"]]
                output_grids = [g.unsqueeze(0).to(self.device) for g in batch["output_grids"]]

                input_seq = sequences[:, :-1]
                target_seq = sequences[:, 1:]

                batch_size = sequences.shape[0]
                num_pairs = len(input_grids) // batch_size

                batch_inputs = []
                batch_outputs = []
                for i in range(min(num_pairs, 1)):
                    for j in range(batch_size):
                        idx = j * num_pairs + i
                        if idx < len(input_grids):
                            batch_inputs.append(input_grids[idx])
                            batch_outputs.append(output_grids[idx])

                if batch_inputs:
                    logits = self.model(batch_inputs, batch_outputs, input_seq)

                    loss = self.criterion(
                        logits.reshape(-1, logits.shape[-1]),
                        target_seq.reshape(-1),
                    )

                    total_loss += loss.item()
                    num_batches += 1

        return total_loss / max(1, num_batches)

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": self.config.to_dict(),
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

    def train(
        self,
        train_dataset: "SyntheticSketcherDataset",
        val_dataset: "SyntheticSketcherDataset",
    ) -> Dict[str, List[float]]:
        """Full training loop."""
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, self.config.model_config.max_seq_len),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, self.config.model_config.max_seq_len),
        )

        history = {"train_loss": [], "val_loss": []}

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch(train_loader)
            history["train_loss"].append(train_loss)

            # Validate
            val_loss = self.validate(val_loader)
            history["val_loss"].append(val_loss)

            print(f"Epoch {epoch + 1}/{self.config.epochs}: "
                  f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(str(output_dir / "sketcher_best.pt"))
                print(f"  -> New best model saved!")

            # Periodic checkpoint
            if (epoch + 1) % self.config.checkpoint_every == 0:
                self.save_checkpoint(str(output_dir / f"sketcher_epoch_{epoch + 1}.pt"))

        # Save final model
        self.save_checkpoint(str(output_dir / "sketcher_final.pt"))

        return history


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train the neural sketcher model")

    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--train-samples", type=int, default=10000, help="Training samples")
    parser.add_argument("--val-samples", type=int, default=1000, help="Validation samples")
    parser.add_argument("--output", type=str, default="models", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is required for training")
        return 1

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create config
    config = SketcherTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_samples=args.train_samples,
        num_val_samples=args.val_samples,
        output_dir=args.output,
    )

    # Create datasets
    print("Creating training dataset...")
    train_dataset = SyntheticSketcherDataset(config.num_train_samples, config)

    print("Creating validation dataset...")
    val_dataset = SyntheticSketcherDataset(config.num_val_samples, config)

    # Create trainer and train
    trainer = SketcherTrainer(config)
    print(f"\nTraining on {trainer.device}")
    print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")

    history = trainer.train(train_dataset, val_dataset)

    # Save history
    output_dir = Path(args.output)
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\nTraining complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Models saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
