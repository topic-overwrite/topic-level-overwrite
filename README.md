## All links are anonymous, some functions may be restricted.


## Prepare <!-- omit in toc -->

1. Install some important packages.

```bash
conda create -n tpr python=3.10 -y
conda activate tpr
pip install -r requirements.txt
```


2. Download Base Model

We recommend downloading the following models:

[llava-1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b)

[llava-next-34b](https://huggingface.co/liuhaotian/llava-v1.6-34b)

[llava-1.5-vision-tower-clip](https://huggingface.co/openai/clip-vit-large-patch14-336)




## Train

**1. Prepare data**

Due to anonymization, we do not provide a dataset link here, only support the data generated from running Data Generation process.


**2. Training**


Run the following command to start fully fine-tuning.

```bash
# you can adjust the hyperparameters for your dataset/model path
bash script/train/llava15_train_main.sh [llava_15_7b_path]  ./tpr_data/generated-dpo-traindata
```

Run the following command to start lora training.

```bash
# you can adjust the hyperparameters for your dataset/model path
bash script/train/llava15_train_lora.sh [llava_15_7b_path]  ./tpr_data/generated-dpo-traindata
```




## Data Generation 

If you prefer to manually generate the dataset rather than using the existing datasets on Hugging Face. 

Follow RLAIF-V work (a famous RLAIF hallucination work), please download supplement model ([Llama-3-8B](meta-llama/Meta-Llama-3-8B-Instruct), (optional) [split model and question transformation model in RLAIF-V repository](https://github.com/RLHF-V/RLAIF-V)) and raw train dataset [RLAIF-V-Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset).


Run the following program.


```bash
# Filter raw data to our format
bash script/data_gen/filter_raw_data.sh [your_downloaded_input_data_folder or openbmb/RLAIF-V-Dataset]

# Run data generate script, you can adjust the hyperparameters for more detailed experiments
# if you need the complete process, you need to iteratively run the data_generation and train processes 5 times, 
# change the 'start_pos'(line 23) and 'end_pos'(line 24) in script/data_gen/data_pipeline_main.sh each iter, generating 4000 different data each iter.
bash script/data_gen/data_pipeline_main.sh   [gpu_num]  [llava_15_7b_path]  [llama_3_8b_path]  [llava_next_34b_path]  [clip_path]  [optional: split_model_path]  [optional: question_transformation_model_path]
```


## Evaluation

During evaluation, object-halbench/mmhal-bench/llava-bench need to be assessed using GPT-3.5/4.

### Object-HalBench

1. Download data from [COCO](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)

2. Download eval supplement model in python

```python
import nltk
nltk.download('wordnet')
nltk.download('punkt')
```

3. Download eval supplement model in terminal

```bash
python -m spacy download en_core_web_trf
```

4. Eval model

```bash
python script/eval/eval_objhal.sh [ckpt_path] [base_path if use lora ckpt else "No"] [YOUR_OPENAI_API_KEY]
```

We default use **gpt-3.5-turbo-0125**, Please replace {YOUR_OPENAI_API_KEY} with a valid OpenAI api-key or directly modify the [13th](https://github.com/topic-overwrite/topic-level-overwrite/blob/main/eval/gpt4_grpc.py#L13) line in eval/gpt4_grpc.py.

### MMHal-Bench

1. Download data from [MMHal-Bench](https://drive.google.com/file/d/1mQyAbeGgRyiVV6qjVkUI1uY_g9E-bDTH/view?usp=sharing).

2. Eval model

```bash
python script/eval/eval_mmhal.sh [ckpt_path] [base_path if use lora ckpt else "No"] [YOUR_OPENAI_API_KEY]
```

We default use **gpt-4-1106-preview**, Please replace {YOUR_OPENAI_API_KEY} with a valid OpenAI api-key or directly modify the [13th](https://github.com/topic-overwrite/topic-level-overwrite/blob/main/eval/gpt4_grpc.py#L13) line in eval/gpt4_grpc.py.

### AMBER

1. Download AMBER [data](https://github.com/junyangwang0410/AMBER/tree/master) and [image](https://drive.google.com/file/d/1MaCHgtupcZUjf007anNl4_MV0o4DjXvl/view?usp=sharing)

2. Download eval supplement model in terminal

```bash
python -m spacy download en_core_web_lg
```

3. Eval model

```bash
python script/eval/eval_amber.sh [ckpt_path] [base_path if use lora ckpt else "No"]
```

### MMSTAR

1. Download data from [MMSTAR](dataset/llava_bench/rule.json).

2. Eval model

```bash
python script/eval/eval_mmstar.sh [ckpt_path] [base_path if use lora ckpt else "No"]
```

### LLaVA-Bench

1. Download data from [LLaVA-Bench](https://huggingface.co/datasets/lmms-lab/llava-bench-in-the-wild).

2. Eval model

```bash
python script/eval/eval_llavabench.sh [ckpt_path] [base_path if use lora ckpt else "No"] [YOUR_OPENAI_API_KEY]
```

We default use **gpt-4-1106-preview**, Please replace {YOUR_OPENAI_API_KEY} with a valid OpenAI api-key or directly modify the [13th](https://github.com/topic-overwrite/topic-level-overwrite/blob/main/eval/gpt4_grpc.py#L13) line in eval/gpt4_grpc.py.



## Acknowledgement

[RLAIF-V](https://github.com/RLHF-V/RLAIF-V): The codebase we built upon.

[LLaVA](https://github.com/haotian-liu/LLaVA): The instruction model and labeler model.