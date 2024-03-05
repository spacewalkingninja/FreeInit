import os, sys
APP_PATH = os.path.dirname(os.path.realpath(__file__))
CDIR = os.getcwd()
os.chdir(APP_PATH)

import argparse
import datetime
import inspect
from omegaconf import OmegaConf

import torch

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationFreeInitPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.util import load_weights
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat

import csv, pdb, glob
import math
from pathlib import Path
from diffusers.training_utils import set_seed


import time, re, json, shutil
import requests, subprocess
import yaml
from yaml.loader import SafeLoader
import io, base64
import argparse

#import gradio as gr
#import spaces

from requests.adapters import HTTPAdapter, Retry

s = requests.Session()

retries = Retry(total=5,
                backoff_factor=0.1,
                status_forcelist=[ 500, 502, 503, 504 ])

s.mount('https://', HTTPAdapter(max_retries=retries))
parser = argparse.ArgumentParser()
parser.add_argument("-mr", "--model_req", 
                    help="DeSOTA Request as yaml file path",
                    type=str)
parser.add_argument("-mru", "--model_res_url",
                    help="DeSOTA API Result URL. Recognize path instead of url for desota tests", # check how is atribuited the test_mode variable in main function
                    type=str)

parser.add_argument("--pretrained_model_path", type=str, default="models/StableDiffusion/stable-diffusion-v1-5",)
parser.add_argument("--inference_config",      type=str, default="configs/inference/inference-v1.yaml")    
parser.add_argument("--config",                type=str, default="configs/text2vid-desota.yaml")

parser.add_argument("--L", type=int, default=16 )
parser.add_argument("--W", type=int, default=128)
parser.add_argument("--H", type=int, default=128)
parser.add_argument("--num_iters", type=int, default=5, help="number of sampling iterations, no freeinit when num_iters=1")
parser.add_argument("--use_fast_sampling", action='store_true')
parser.add_argument("--save_intermediate", action='store_true')
parser.add_argument("--use_fp16", action='store_true')

parser.add_argument("-deb", "--debug",
                    help="DeSOTAdebug", # check how is atribuited the test_mode variable in main function
                    type=int, default=0)
parser.add_argument("-p", "--dprompt",
                    help="fast prompt for debug", # check how is atribuited the test_mode variable in main function
                    type=str, default='fruit bananas space station orbit earth')

DEBUG = False

# DeSOTA Funcs [START]
#   > Import DeSOTA Scripts
from desota import detools
#   > Grab DeSOTA Paths
USER_SYS = detools.get_platform()

#   > USER_PATH
if USER_SYS == "win":
    path_split = str(APP_PATH).split("\\")
    desota_idx = [ps.lower() for ps in path_split].index("desota")
    USER=path_split[desota_idx-1]
    USER_PATH = "\\".join(path_split[:desota_idx])
elif USER_SYS == "lin":
    path_split = str(APP_PATH).split("/")
    desota_idx = [ps.lower() for ps in path_split].index("desota")
    USER=path_split[desota_idx-1]
    USER_PATH = "/".join(path_split[:desota_idx])
DESOTA_ROOT_PATH = os.path.join(USER_PATH, "Desota")
TMP_PATH = os.path.join(APP_PATH, "TMP")
# DeSOTA Funcs [END]

def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    
    #time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    #savedir = f"outputs/{Path(args.config).stem}-{time_str}"
    savedir = TMP_PATH
    os.makedirs(savedir,  exist_ok=True)
    #Clean everything for rerunnnnnn
    for filename in os.listdir(TMP_PATH):
        file_path = os.path.join(TMP_PATH, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    config  = OmegaConf.load(args.config)
    samples = []
    
    sample_idx = 0

    # set global seed
    set_seed(42)




    '''
    return codes:
    0 = SUCESS
    1 = INPUT ERROR
    2 = OUTPUT ERROR
    3 = API RESPONSE ERROR
    9 = REINSTALL MODEL (critical fail)
    '''
    # Time when grabed
    _report_start_time = time.time()
    start_time = int(_report_start_time)

    #---INPUT---# TODO (PRO ARGS)
    #---INPUT---#
    req_text = ''
    if args.debug == 0:
    # DeSOTA Model Request
        model_request_dict = detools.get_model_req(args.model_req)

        # API Response URL
        result_id = args.model_res_url
        
        # TARGET File Path
        dir_path = os.path.dirname(os.path.realpath(__file__))
        #out_filepath = os.path.join(dir_path, f"txt-to-img-{start_time}.png")
        out_urls = detools.get_url_from_str(result_id)
        if len(out_urls)==0:
            test_mode = True
            report_path = result_id
        else:
            test_mode = False
            send_task_url = out_urls[0]

        # Get html file
        req_text = detools.get_request_text(model_request_dict)

        if isinstance(req_text, list):
            req_text = ' '.join(req_text)

    
    if req_text or args.debug == 1:
        if args.debug == 1:
            test_mode=True
            report_path='test.json'
            req_text = args.dprompt
            model_request_dict = {
                'input_args':{}
                }
        default_config = {
        #"prompt": str(args.text_prompt),
        "width":128,
        "height":128,
        "length":16,
        "num_inference_steps":20,
        "num_smoothing_steps":3,
        "guidance_scale":7.5,
        "temporal_start":0.25,
        "temporal_stop":0.25,
        "seed":96642
        }
        targs = {}
        if 'model_args' in model_request_dict['input_args']:
            targs = model_request_dict['input_args']['model_args']

        if 'prompt' in targs:
            if targs['prompt'] == '$initial-prompt$':
                targs['prompt'] = req_text
        else:
            targs['prompt'] = req_text
        confyconf = default_config | targs

        confy_filter_params = {
            "method": "butterworth",
            "n": int(confyconf['num_smoothing_steps']),
            "d_s": float(confyconf['temporal_start']),
            "d_t": float(confyconf['temporal_stop'])
        }

        for model_idx, (config_key, model_config) in enumerate(list(config.items())):
            
            motion_modules = model_config.motion_module
            motion_modules = [motion_modules] if isinstance(motion_modules, str) else list(motion_modules)
            for motion_module in motion_modules:
                inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))

                ### >>> create validation pipeline >>> ###
                tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
                text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
                vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")            
                unet         = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))

                if is_xformers_available(): unet.enable_xformers_memory_efficient_attention()
                else: assert False

                pipeline = AnimationFreeInitPipeline(
                    vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                    scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
                ).to("cuda")

                pipeline = load_weights(
                    pipeline,
                    # motion module
                    motion_module_path         = motion_module,
                    motion_module_lora_configs = model_config.get("motion_module_lora_configs", []),
                    # image layers
                    dreambooth_model_path      = model_config.get("dreambooth_path", ""),
                    lora_model_path            = model_config.get("lora_model_path", ""),
                    lora_alpha                 = model_config.get("lora_alpha", 0.8),
                ).to("cuda")

                # (freeinit) initialize frequency filter for noise reinitialization -------------
                pipeline.init_filter(
                    width               = confyconf['width'],
                    height              = confyconf['height'],
                    video_length        = confyconf['length'],
                    filter_params       = confy_filter_params,
                )
                # -------------------------------------------------------------------------------

                prompts      = [confyconf['prompt']]
                n_prompts    = ["worst quality, low quality, nsfw, logo, watermark"]
                #n_prompts    = list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt
                
                #random_seeds = model_config.get("seed", [-1])
                random_seeds = confyconf['seed']
                random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
                random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
                
                config[config_key].random_seed = []
                for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):
                    
                    # manually set random seed for reproduction
                    # if random_seed != -1: torch.manual_seed(random_seed)
                    if random_seed != -1: set_seed(random_seed)
                    else: torch.seed()
                    config[config_key].random_seed.append(torch.initial_seed())
                    
                    print(f"current seed: {torch.initial_seed()}")
                    print(f"sampling {prompt} ...")
                    save_prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
                    
                    sample = pipeline(
                        prompt,
                        negative_prompt     = n_prompt,
                        num_inference_steps = confyconf['num_inference_steps'],
                        guidance_scale      = confyconf['guidance_scale'],
                        width               = confyconf['width'],
                        height              = confyconf['height'],
                        video_length        = confyconf['length'],
                        num_iters = confyconf['num_smoothing_steps'],
                        use_fast_sampling = args.use_fast_sampling,
                        save_intermediate = True,
                        save_dir = TMP_PATH,
                        save_name = f"{sample_idx}-{save_prompt}",
                        use_fp16            = True #args.use_fp16
                    ).videos
                    samples.append(sample)

                    save_videos_grid(sample, os.path.join(TMP_PATH, f"{sample_idx}-{save_prompt}.gif"))
                    #print(f"save to {savedir}/sample/{save_prompt}.gif")
                    
                    sample_idx += 1

        samples = torch.concat(samples)
        out_filepath = os.path.join(TMP_PATH, f"sample.gif")
        save_videos_grid(samples, out_filepath, n_rows=4)

        OmegaConf.save(config,  os.path.join(TMP_PATH, f"{savedir}/config.yaml"))
     
    if not os.path.isfile(out_filepath):
        print(f"[ ERROR ] -> DeSOTA FreeAnimateInit API Output ERROR: No results can be parsed for this request")
        os.chdir(CDIR)
        exit(2)
        
    #print(f"[ INFO ] -> Response:\n{json.dumps(r, indent=2)}")
    
    if test_mode:
        if not report_path.endswith(".json"):
            report_path += ".json"
        with open(report_path, "w") as rw:
            json.dump(
                {
                    "Model Result Path": out_filepath,
                    "Processing Time": time.time() - _report_start_time
                },
                rw,
                indent=2
            )
        detools.user_chown(report_path)
        #detools.user_chown(outfile)
        print(f"Path to report:\n\t{report_path}")
    else:
        files = []
        with open(out_filepath, 'rb') as fr:
            files.append(('upload[]', fr))
            # DeSOTA API Response Post
            send_task = s.post(url = send_task_url, files=files)
            print(f"[ INFO ] -> DeSOTA API Upload Res:\n{json.dumps(send_task.json(), indent=2)}")
        # Delete temporary file
        os.remove(out_filepath)

        if send_task.status_code != 200:
            print(f"[ ERROR ] -> DeSOTA SD.Next API Post Failed (Info):\nfiles: {files}\nResponse Code: {send_task.status_code}")
            os.chdir(CDIR)
            exit(3)
    
    print("TASK OK!")
    os.chdir(CDIR)
    exit(0)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    os.chdir(CDIR)
    exit(1)