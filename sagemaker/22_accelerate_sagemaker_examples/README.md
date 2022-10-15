# accelerate-aws-sagemaker
Examples showcasing AWS SageMaker integration of 🤗 Accelerate. Just give the `accelerate config` and do `accelerate launch` 🚀. It's as simple as that!

1. Set up the accelerate config by running `accelerate config --config_file accelerate_config.yaml` and answer the SageMaker questions.

2. Below is a sample config which is using aws `profile` to launch training job using 🤗 SageMaker estimator. It also has the `iam_role_name` which has the needed SageMaker permissions specified. In this config it is replaced `xxxxx` as user needs to specify it based on their corresponding AWS setup.

```yaml
base_job_name: accelerate-sagemaker-1
compute_environment: AMAZON_SAGEMAKER
distributed_type: DATA_PARALLEL
ec2_instance_type: ml.p3.16xlarge
iam_role_name: xxxxx
image_uri: null
mixed_precision: fp16
num_machines: 1
profile: xxxxx
py_version: py38
pytorch_version: 1.10.2
region: us-east-1
transformers_version: 4.17.0
use_cpu: false
```
3. One can specify a custom docker image instead of Official 🤗 DLCs through the accelerate config questionnaire. When this isn't provided, the latest Official 🤗 DLC will be used.

4. Support for input channels pointing to S3 data locations via TSV file, e.g., below are the contents of sagemaker_inputs.tsv whose location is given as part of accelerate config setup.
```tsv
channel_name	data_location
train	s3://sagemaker-sample/samples/datasets/imdb/train
test	s3://sagemaker-sample/samples/datasets/imdb/test
```

5. Support for SageMaker metrics logging via TSV file, e.g., below are the contents of the sagemaker_metrics_definition.tsv whose location is given as part of accelerate config setup.
```tsv
metric_name	metric_regex
accuracy	'accuracy': ([0-9.]+)
f1	'f1': ([0-9.]+)
```

6. Example of accelerate config with above features setup [XXXXX values are AWS account specific]:
```yaml
base_job_name: accelerate-sagemaker-1
compute_environment: AMAZON_SAGEMAKER
distributed_type: DATA_PARALLEL
ec2_instance_type: ml.p3.16xlarge
iam_role_name: XXXXX
image_uri: 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:1.8.1-transformers4.10.2-gpu-py36-cu111-ubuntu18.04
mixed_precision: fp16
num_machines: 1
profile: XXXXX
py_version: py38
pytorch_version: 1.10.2
region: us-east-1
sagemaker_inputs_file: sagemaker_inputs.tsv
sagemaker_metrics_file: sagemaker_metrics_definition.tsv
transformers_version: 4.17.0
use_cpu: false
```
7. Put `requirements.txt` with all the needed libraries for running the training script.

8. Running `text-classification` example using s3 datasets (from the root directory):
```bash
cd src/text-classification
bash launch.sh
```
The contents of launch.sh
```bash
accelerate launch  --config_file accelerate_config.yaml train_using_s3_data.py \
    --mixed_precision "fp16"
```
Output logs:
```bash
...

[1,mpirank:0,algo-1]<stdout>:algo-1:79:1300 [0] NCCL INFO Launch mode Parallel
[1,mpirank:0,algo-1]<stderr>:INFO:root:Reducer buckets have been rebuilt in this iteration.
[1,mpirank:3,algo-1]<stderr>:INFO:root:Reducer buckets have been rebuilt in this iteration.
[1,mpirank:1,algo-1]<stderr>:INFO:root:Reducer buckets have been rebuilt in this iteration.
[1,mpirank:2,algo-1]<stderr>:INFO:root:Reducer buckets have been rebuilt in this iteration.
[1,mpirank:6,algo-1]<stderr>:INFO:root:Reducer buckets have been rebuilt in this iteration.
[1,mpirank:5,algo-1]<stderr>:INFO:root:Reducer buckets have been rebuilt in this iteration.
[1,mpirank:7,algo-1]<stderr>:INFO:root:Reducer buckets have been rebuilt in this iteration.
[1,mpirank:4,algo-1]<stderr>:INFO:root:Reducer buckets have been rebuilt in this iteration.
[1,mpirank:0,algo-1]<stdout>:epoch 0: {'accuracy': 0.6838235294117647, 'f1': 0.8122270742358079}
[1,mpirank:0,algo-1]<stdout>:epoch 1: {'accuracy': 0.7205882352941176, 'f1': 0.8256880733944955}
[1,mpirank:0,algo-1]<stdout>:epoch 2: {'accuracy': 0.75, 'f1': 0.838095238095238}
2022-09-21 13:21:05,187 sagemaker-training-toolkit INFO     Waiting for the process to finish and give a return code.
2022-09-21 13:21:05,188 sagemaker-training-toolkit INFO     Done waiting for a return code. Received 0 from exiting process.
2022-09-21 13:21:05,188 sagemaker-training-toolkit INFO     Reporting training SUCCESS
```


9. Running `seq2seq` example:
```bash
cd src/seq2seq
bash launch.sh
```
The contents of launch.sh
```bash
accelerate launch --config_file accelerate_config.yaml run_seq2seq_no_trainer.py \
    --dataset_name "smangrul/MuDoConv" \
    --max_source_length 128 \
    --source_prefix "chatbot: " \
    --max_target_length 64 \
    --val_max_target_length 64 \
    --val_min_target_length 20 \
    --n_val_batch_generations 5 \
    --n_train 10000 \
    --n_val 1000 \
    --pad_to_max_length True\
    --num_beams 10 \
    --model_name_or_path "facebook/blenderbot-400M-distill" \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 1e-6 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 1 \
    --num_warmup_steps 100 \
    --output_dir "/opt/ml/model" \
    --seed 25 \
    --logging_steps 100 \
    --report_name "blenderbot_400M_finetuning"
```
Output logs:
```
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:37:39 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: smddp
[1,mpirank:0,algo-1]<stderr>:Num processes: 8
[1,mpirank:0,algo-1]<stderr>:Process index: 0
[1,mpirank:0,algo-1]<stderr>:Local process index: 0
[1,mpirank:0,algo-1]<stderr>:Device: cuda:0
[1,mpirank:0,algo-1]<stderr>:

...

[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:38:31 - INFO - __main__ - ***** Running training *****
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:38:31 - INFO - __main__ -   Num examples = 10000
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:38:31 - INFO - __main__ -   Num Epochs = 1
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:38:31 - INFO - __main__ -   Instantaneous batch size per device = 16
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:38:31 - INFO - __main__ -   Total train batch size (w. parallel, distributed & accumulation) = 128
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:38:31 - INFO - __main__ -   Gradient Accumulation steps = 1
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:38:31 - INFO - __main__ -   Total optimization steps = 79

...

[1,mpirank:0,algo-1]<stderr>:#015100%|██████████| 79/79 [00:19<00:00,  4.79it/s]
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:38:50 - INFO - __main__ - Epoch 0 training took 19.50162172317505 seconds
[1,mpirank:0,algo-1]<stdout>:starting evaluation
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:38:55 - INFO - __main__ - printing few sample generations and corresponding labels from eval set
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:38:55 - INFO - __main__ - prompt | generated | label
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:38:55 - INFO - __main__ - chatbot:your persona: i work as an electrician. i always sleep 8 hours a day. </s>  <s> Which level are you at?</s> |  I'm at the top of the ladder. I work for an electrical company. |  I received on-the-job training when i first started
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:38:55 - INFO - __main__ - chatbot:your persona: i work as an electrician. i always sleep 8 hours a day. </s>  <s> Which level are you at? </s> <s> I received on-the-job training when i first started </s> <s> Thats great! How long have you been doing this work? </s> |  I've been working as an Electrician for about 5 years now. It's a great job. |  For a good number of years now.

... 

[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:39:02 - INFO - __main__ - chatbot:your persona: i'm a painter and love to create art. i'm a talented singer and have won several competitions. </s>  <s> I love the Doors! They have such a unique sound. Do you have a favorite Doors song?</s> |  My favorite song of theirs is "When I Was Your Man". What's yours? |  It's a tie between People are strange and Love me two times. What's your favorite?
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:39:02 - INFO - __main__ - chatbot:your persona: i'm a painter and love to create art. i'm a talented singer and have won several competitions. </s>  <s> I think my favorite is Love Street. It has such a haunting melody. Have you heard that one?</s> |  No, I haven't. I'll have to check it out. What genre is it? |  Yeah, it's a pretty great song, Jim Morrison was just an amazing songwriter, I aspire to make songs as good as his one day.

...

[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:39:17 - INFO - __main__ - {'bleu': 1.7067114414104911}
[1,mpirank:0,algo-1]<stdout>:evaluation completed
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:39:17 - INFO - __main__ - Epoch 0 evaluation took 24.294514417648315 seconds
[1,mpirank:0,algo-1]<stderr>:Configuration saved in /opt/ml/model/config.json
[1,mpirank:0,algo-1]<stderr>:Model weights saved in /opt/ml/model/pytorch_model.bin
[1,mpirank:0,algo-1]<stderr>:tokenizer config file saved in /opt/ml/model/tokenizer_config.json
[1,mpirank:0,algo-1]<stderr>:Special tokens file saved in /opt/ml/model/special_tokens_map.json
[1,mpirank:0,algo-1]<stderr>:#015100%|██████████| 79/79 [00:47<00:00,  1.65it/s]
2022-09-21 13:39:27,753 sagemaker-training-toolkit INFO     Waiting for the process to finish and give a return code.
2022-09-21 13:39:27,753 sagemaker-training-toolkit INFO     Done waiting for a return code. Received 0 from exiting process.
2022-09-21 13:39:27,754 sagemaker-training-toolkit INFO     Reporting training SUCCESS
```


