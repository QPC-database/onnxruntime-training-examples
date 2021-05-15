import os
import requests
import sys
import shlex
import statistics
from datetime import datetime

# AzureML libraries
import azureml.core
from azureml.core import Experiment, Workspace, Datastore, Run, Environment
from azureml.core.compute import ComputeTarget, AmlCompute, AksCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import ScriptRunConfig
from azureml.core.runconfig import PyTorchConfiguration

# Check core SDK version number
print("SDK version:", azureml.core.VERSION)

ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')

# Create the compute cluster
gpu_cluster_name = "v100" 

# Verify that the cluster doesn't exist already
try:
    gpu_compute_target = ComputeTarget(workspace=ws, name=gpu_cluster_name)
    print('Found existing compute target.')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_ND40rs_v2', min_nodes=0, max_nodes=7)
    
    # create the cluster
    gpu_compute_target = ComputeTarget.create(ws, gpu_cluster_name, compute_config)
    gpu_compute_target.wait_for_completion(show_output=True)


hf_models = ['bert-large', 'distilbert-base', 'gpt2', 'bart-large', 't5-large']

run_configs = ['pt-fp16', 'ort', 'ds_s0', 'ds_s0_ort', 'ds_s1', 'ds_s1_ort']

torch_distributed_args = "python -m torch.distributed.launch --nproc_per_node 8 --use_env "

model_batchsize = {
    "bert-large" : '8',
    "distilbert-base" : '32',
    "gpt2" : '8',
    "bart-large" : '16',
    "t5-large" : '16'
}

run_scripts = {
    "bert-large" : 'run_mlm.py',
    "distilbert-base" : 'run_mlm.py',
    "gpt2" : 'run_clm.py',
    "bart-large" : 'run_translation.py',
    "t5-large" : 'run_translation.py'
}

base_args = {
    "bert-large" : ['--model_name_or_path', 'bert-large-uncased', '--dataset_name', 'wikitext', '--dataset_config_name', 'wikitext-2-raw-v1', '--do_train', '--max_steps', 200, '--logging_steps', 200, '--output_dir', '/tmp/test-mlm-bbu', '--overwrite_output_dir', '--per_device_train_batch_size', 8, '--fp16'],
    "distilbert-base" : ['--model_name_or_path', 'distilbert-base-uncased', '--dataset_name', 'wikitext', '--dataset_config_name', 'wikitext-2-raw-v1', '--do_train', '--max_steps', 200, '--logging_steps', 200, '--output_dir', '/tmp/test-mlm-bbu', '--overwrite_output_dir', '--per_device_train_batch_size', 32, '--fp16'],
    "gpt2" : ['--model_name_or_path', 'gpt2', '--dataset_name', 'wikitext', '--dataset_config_name', 'wikitext-2-raw-v1', '--do_train', '--label_smoothing', 0.1, '--max_steps', 200, '--logging_steps', 200, '--overwrite_output_dir', '--output_dir', '/tmp/test-clm', '--per_device_train_batch_size', 8, '--fp16'],
    "bart-large" : ['--dataset_name', 'wmt16', '--dataset_config', 'ro-en', '--model_name_or_path', 'facebook/bart-large', '--output_dir', '/tmp/tst-translation', '--do_train', '--label_smoothing', 0.1, '--logging_steps', 200, '--overwrite_output_dir', '--per_device_train_batch_size', 16, '--predict_with_generate', '--source_lang', 'en', '--target_lang', 'ro', '--warmup_steps', 5, '--fp16', '--max_steps', 200],
    "t5-large" : ['--source_prefix', 'translate English to Romanian:', '--dataset_name', 'wmt16', '--dataset_config', 'ro-en', '--model_name_or_path', 't5-large', '--output_dir', '/tmp/tst-translation', '--do_train', '--label_smoothing', 0.1, '--logging_steps', 200, '--overwrite_output_dir', '--per_device_train_batch_size', 16, '--predict_with_generate', '--source_lang', 'en', '--target_lang', 'ro', '--warmup_steps', 5, '--fp16', '--max_steps', 200],
}

config_args = {
    "pt-fp16" : [],
    "ort" : ['--ort'],
    "ds_s0" : ['--deepspeed', 'ds_config_zero_0.json'],
    "ds_s1" : ['--deepspeed', 'ds_config_zero_1.json'],
    "ds_s0_ort" : ['--ort', '--deepspeed', 'ds_config_zero_0.json'],
    "ds_s1_ort" : ['--ort', '--deepspeed', 'ds_config_zero_1.json'],
}

hf_ort_env = Environment.from_dockerfile(name='hf-ort-dockerfile', dockerfile='Dockerfile.dockerfile')
hf_ort_env.register(ws).build(ws).wait_for_completion()

distr_config = PyTorchConfiguration(process_count=8, node_count=1)

experiment_name = 'hf-ortmodule-'
for hf_model in hf_models:
    model_experiment_name = experiment_name + hf_model
    model_run_args_base = base_args[hf_model]
    model_run_script = run_scripts[hf_model]
    # Create experiment for model
    model_experiment = Experiment(ws, name=model_experiment_name)

    for run_config in run_configs:
        model_run_args_config = model_run_args_base + config_args[run_config]
        # create script run config for the model+config
        model_run_config = ScriptRunConfig(source_directory='.',
            script=model_run_script,
            arguments=model_run_args_config,
            compute_target=gpu_compute_target,
            environment=hf_ort_env,
            distributed_job_config=distr_config)

        print("Submitting run for model: ", hf_model)
        print("Submitting run for config: ", run_config)
        run = model_experiment.submit(model_run_config)
        run.add_properties({'model' : hf_model, 'config' : run_config, 'bs' : model_batchsize[hf_model], 'gpus' : '8'})
