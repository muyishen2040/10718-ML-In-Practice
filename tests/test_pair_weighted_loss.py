import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from src.verification.pair_reader import PairReaderConfig, weighted_verdict_loss


def test_corrected_class_weight_changes_singleton_loss_and_gradient():
    logits = torch.tensor([[.2, -.1, .4, 0.]], requires_grad=True)
    y, weights = torch.tensor([3]), torch.tensor([.5, .5, 1., 4.])
    plain = torch.nn.functional.cross_entropy(logits, y)
    legacy = weighted_verdict_loss(logits, y, weights, "batch_weight_mean")
    corrected = weighted_verdict_loss(logits, y, weights, "sample_mean")
    torch.testing.assert_close(legacy, plain)
    torch.testing.assert_close(corrected, 4 * plain)
    old_grad = torch.autograd.grad(legacy, logits, retain_graph=True)[0]
    new_grad = torch.autograd.grad(corrected, logits)[0]
    torch.testing.assert_close(new_grad, 4 * old_grad)


def test_weighted_objective_is_invariant_to_equal_microbatch_accumulation():
    torch.manual_seed(718)
    logits = torch.randn(8, 4, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    weights = torch.tensor([.5, .25, 2., 4.])
    full = weighted_verdict_loss(logits, labels, weights)
    micro = sum(weighted_verdict_loss(logits[i:i + 2], labels[i:i + 2], weights) for i in range(0, 8, 2)) / 4
    torch.testing.assert_close(full, micro)
    torch.testing.assert_close(torch.autograd.grad(full, logits, retain_graph=True)[0], torch.autograd.grad(micro, logits)[0])
    legacy_micro = sum(weighted_verdict_loss(logits[i:i + 2], labels[i:i + 2], weights, "batch_weight_mean") for i in range(0, 8, 2)) / 4
    assert not torch.isclose(full, legacy_micro)


def test_old_config_retains_legacy_loss_semantics():
    assert PairReaderConfig().loss_normalization == "batch_weight_mean"
    with pytest.raises(ValueError):
        weighted_verdict_loss(torch.zeros(1, 4), torch.tensor([0]), torch.ones(4), "unknown")


def test_real_offline_extractive_model_offsets_and_both_question_outputs(tmp_path):
    from transformers import BertConfig, BertForQuestionAnswering, BertTokenizerFast
    from src.retrieval.answer_followup import ExtractiveReader, pack_answers
    directory = tmp_path / "qa"
    directory.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "where", "when", "?", "the", "event", "was", "in", "london", "on", "monday", "."]
    (directory / "vocab.txt").write_text("\n".join(vocab) + "\n")
    tokenizer = BertTokenizerFast(vocab_file=str(directory / "vocab.txt"))
    tokenizer.save_pretrained(directory)
    model = BertForQuestionAnswering(BertConfig(vocab_size=len(vocab), hidden_size=16,
        num_hidden_layers=1, num_attention_heads=2, intermediate_size=24))
    model.save_pretrained(directory)
    reader = ExtractiveReader(str(directory), None, "cpu", 2, 64)
    candidates = [{"passage_id": "p", "text": "The event was in London on Monday."}]
    output = reader(["Where?", "When?"], candidates)
    assert len(output) == 1 and len(output[0]) == 2
    assert [a["question_index"] for a in output[0]] == [0, 1]
    for answer in output[0]:
        assert answer["answer"] in candidates[0]["text"]
    packed = pack_answers(candidates[0], output[0])
    assert "Where?" in packed["text"] and "When?" in packed["text"]
