# Processors

Processor is a basic building block that is responsible on converting rows from a huggingface dataset to a conversation in the **ChatML** format.

All processors extends the [BaseProcessor](base.py) class and must define the following:

* `dataset_id` (attribute) - this is a class attribute that should point to the dataset id of the dataset on the HuggingFace hub. Click [here](https://huggingface.co/docs/hub/datasets-adding) for a detailed guide on uploading datasets.
* `to_chat_template()` (function) - this is the function that transforms a row in the dataset (`example`) into a conversation in the ChatML format. Notice that you have to implement 2 versions of this function:
  1. When `assistant_response = True` - typically used for training or testing, when you have the labels for the input example.
  2. When `assistant_response = False` - used for actual inference when the labels are not known in advance.

Once you define a processor for your dataset, don't forget to import it in the [\_\_init\_\_.py](__init__.py) file to register it to the [ProcessorRegistry](base.py), and enable automatic integration to [finetune_sft.py](src/training/finetune_sft.py).
