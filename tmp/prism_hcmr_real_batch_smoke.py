import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recommender.prism.config import get_config
from src.recommender.prism.dataset import create_datasets
from src.recommender.prism.dataloader import collate_fn
from src.recommender.prism.model import create_model


def main():
    torch.manual_seed(0)

    config = get_config(
        "beauty",
        use_multimodal_fusion=True,
        use_purified_predictor=True,
        use_item_layer_emb=True,
        use_trie_constraints=True,
        use_adaptive_temperature=False,
        batch_size=2,
        eval_batch_size=2,
        num_workers=0,
        warmup_steps=2500,
        hcmr_memory_topk=6,
        hcmr_recency_bias=0.05,
        purified_predictor_warmup_steps=3000,
    )

    train_dataset, _, _, _ = create_datasets(
        sequence_data_dir=config["data"].sequence_data_path,
        semantic_mapping_path=config["data"].semantic_mapping_path,
        max_len=config["data"].max_seq_length,
        codebook_size=config["model"].codebook_size,
        num_layers=config["model"].num_code_layers,
        pad_token_id=config["model"].pad_token_id,
        model_config=config["model"],
        codebook_sizes=config["model"].__dict__.get("codebook_sizes", None),
        purified_content_path=config["data"].purified_content_path,
        purified_collab_path=config["data"].purified_collab_path,
        codebook_zq_path=config["data"].codebook_zq_path,
        use_multimodal=True,
    )

    batch = collate_fn([train_dataset[0], train_dataset[1]], pad_token_id=config["model"].pad_token_id)
    model = create_model(config["model"], config["training"])
    model.current_step = 1500
    model.train()

    output = model(
        input_ids=batch["history"],
        attention_mask=batch["attention_mask"],
        labels=batch["target"],
        purified_content=batch["history_purified_content"],
        purified_collab=batch["history_purified_collab"],
        codebook_zq=batch["history_codebook_zq"],
        history_item_ids=batch["history_item_ids"],
        target_z_clean=batch["target_z_clean"],
        item_ids=batch["target_item_id"],
        return_dict=True,
    )
    loss = output["loss"]
    loss.backward()

    memory_grad = model.memory_builder.memory_item_score.weight.grad
    pool_grad = model.memory_builder.item_pool.weight.grad
    predictor_grad = model.purified_predictor.predictor[0].weight.grad
    assert memory_grad is not None and torch.isfinite(memory_grad).all()
    assert pool_grad is not None and torch.isfinite(pool_grad).all()
    assert predictor_grad is not None and torch.isfinite(predictor_grad).all()

    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_ids=batch["history"],
            attention_mask=batch["attention_mask"],
            num_beams=2,
            max_length=config["model"].num_code_layers + 1,
            purified_content=batch["history_purified_content"],
            purified_collab=batch["history_purified_collab"],
            codebook_zq=batch["history_codebook_zq"],
            history_item_ids=batch["history_item_ids"],
        )

    print(
        {
            "loss": float(loss.item()),
            "main_loss": float(output["main_loss"].item()),
            "pred_loss": float(output["pred_loss"]),
            "pred_weight": float(output["pred_weight"]),
            "memory_tokens_per_sample": output["memory_stats"]["memory_tokens_per_sample"],
            "memory_items_per_sample": output["memory_stats"]["memory_items_per_sample"],
            "memory_selected_score_mean": output["memory_stats"]["memory_selected_score_mean"],
            "gate_q_mean": output["memory_stats"]["memory_gate_q_mean"],
            "gate_t_mean": output["memory_stats"]["memory_gate_t_mean"],
            "gate_c_mean": output["memory_stats"]["memory_gate_c_mean"],
            "memory_score_grad_l1": float(memory_grad.abs().sum().item()),
            "pool_grad_l1": float(pool_grad.abs().sum().item()),
            "predictor_grad_l1": float(predictor_grad.abs().sum().item()),
            "generated_shape": tuple(generated.shape),
        }
    )


if __name__ == "__main__":
    main()
