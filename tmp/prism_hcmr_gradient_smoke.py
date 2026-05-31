import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recommender.prism.config import ModelConfig, TrainingConfig
from src.recommender.prism.model import create_model


def main():
    torch.manual_seed(0)

    model_cfg = ModelConfig(
        num_layers=2,
        num_decoder_layers=2,
        d_model=32,
        d_ff=64,
        num_heads=1,
        d_kv=32,
        num_code_layers=3,
        codebook_size=16,
    )
    model_cfg.set_vocab_size(64)

    train_cfg = TrainingConfig(
        use_multimodal_fusion=True,
        use_purified_predictor=True,
        purified_dim=8,
        codebook_dim=4,
        use_item_layer_emb=True,
        batch_size=2,
        eval_batch_size=2,
        num_epochs=1,
    )

    model = create_model(model_cfg, train_cfg)
    model.train()
    model.current_step = 1_500

    batch_size = 2
    max_items = 4
    num_layers = model_cfg.num_code_layers
    seq_len = max_items * num_layers

    input_ids = torch.randint(1, model_cfg.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = torch.randint(1, model_cfg.vocab_size, (batch_size, num_layers))
    purified_content = torch.randn(batch_size, max_items, train_cfg.purified_dim)
    purified_collab = torch.randn(batch_size, max_items, train_cfg.purified_dim)
    codebook_zq = torch.randn(batch_size, max_items, train_cfg.codebook_dim)
    history_item_ids = torch.tensor(
        [[1, 2, 3, 4], [5, 6, 0, 0]],
        dtype=torch.long,
    )
    target_z_clean = torch.randn(batch_size, train_cfg.purified_dim * 2)
    target_item_ids = [7, 8]

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        purified_content=purified_content,
        purified_collab=purified_collab,
        codebook_zq=codebook_zq,
        history_item_ids=history_item_ids,
        target_z_clean=target_z_clean,
        item_ids=target_item_ids,
        return_dict=True,
    )

    loss = output["loss"]
    loss.backward()

    embedding_grad = model.model.get_input_embeddings().weight.grad
    memory_grad = model.memory_builder.q_proj[0].weight.grad
    predictor_grad = model.purified_predictor.predictor[0].weight.grad

    assert embedding_grad is not None and torch.isfinite(embedding_grad).all()
    assert memory_grad is not None and torch.isfinite(memory_grad).all()
    assert predictor_grad is not None and torch.isfinite(predictor_grad).all()
    assert embedding_grad.abs().sum().item() > 0
    assert memory_grad.abs().sum().item() > 0
    assert predictor_grad.abs().sum().item() > 0

    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=2,
            max_length=num_layers + 1,
            purified_content=purified_content,
            purified_collab=purified_collab,
            codebook_zq=codebook_zq,
            history_item_ids=history_item_ids,
        )
    assert generated.shape[1] == num_layers + 1

    print(
        {
            "loss": float(loss.item()),
            "main_loss": float(output["main_loss"].item()),
            "pred_loss": float(output["pred_loss"]),
            "pred_weight": float(output["pred_weight"]),
            "memory_tokens_per_sample": output["memory_stats"]["memory_tokens_per_sample"],
            "memory_items_per_sample": output["memory_stats"]["memory_items_per_sample"],
            "embedding_grad_l1": float(embedding_grad.abs().sum().item()),
            "memory_grad_l1": float(memory_grad.abs().sum().item()),
            "predictor_grad_l1": float(predictor_grad.abs().sum().item()),
            "generated_shape": tuple(generated.shape),
        }
    )


if __name__ == "__main__":
    main()
