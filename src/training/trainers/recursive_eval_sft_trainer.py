import torch
from trl import SFTTrainer

class RecursiveEvalSFTTrainer(SFTTrainer):
    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        # Build prompt from inputs (e.g., decode or use input_ids directly)
        input_ids = inputs["input_ids"]
        attn_mask = inputs.get("attention_mask")

        # 1) Do your recursive loop
        cur_ids = input_ids
        for _ in range(8):  # max rounds
            out = model.generate(
                input_ids=cur_ids,
                attention_mask=attn_mask,
                max_new_tokens=8192,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
            )
            # check your condition on decoded text / logits
            text = self.tokenizer.batch_decode(out, skip_special_tokens=True)
            if your_condition(text):
                final = out
                break
            # optionally append something and continue
            cur_ids = out

        # 2) Package outputs the way Trainer expects
        # loss can be None; labels come from batch if present
        labels = inputs.get("labels")
        return (None, final.cpu(), labels.cpu() if labels is not None else None)
