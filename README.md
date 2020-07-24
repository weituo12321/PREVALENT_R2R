# PREVALENT_R2R
Apply PREVALENT pretrained model on R2R task. I am clearing the redundant code and classes. But you should still be able to run the code now.   

#### Requirements  

OS: Ubuntu
docker image: vlnres/mattersim:v5  
  


1 install docker and set nvidia-docker
   
-  To install docker please check [here](https://docs.docker.com/engine/install/ubuntu/)
  
- To setup docker to use GPU run:
`sh nvidia-container-runtime-script.sh`

2 create container  

- To pull the image:
```
docker pull vlnres/mattersim:v5
```

- To create the container:
```
docker run -it --gpus 1 --volume "your_work_directory":/root/mount/Matterport3DSimulator vlnres/mattersim:v5
``` 


3 Set up (for some missing dependencies)
```
docker start “your container id or container name”
docker exec -it “your container id or container name”  /bin/bash     
cd /root/mount/Matterport3DSimulator       
pip install --user pytorch-transformers==1.2.0    
pip install --user tensorboardX  
```



#### Train Agent for R2R  
- 1 We follow the same training schedule as [here](https://github.com/airsplay/R2R-EnvDrop). You can train your own speaker and initial back translation agent. Alternatively, you can use provided [speaker](https://drive.google.com/file/d/1vOpHC6NNO5T4j0r0qu1dhEC1ComkYsxS/view?usp=sharing) and [initial agent](https://drive.google.com/file/d/12NZmDGgcoptj6tYK68ceuZnImCbRWqzB/view?usp=sharing). 
- 2 Make sure you already put pretrained_model under `./pretrained_hug_models/dicadd`, initial agent under `./previous_btbert_agent` and trained speaker under  `snap/speaker/`
- 3 Run the following example command (change the directory name accordingly)
  

```
CUDA_VISIBLE_DEVICES=0 python r2r_src/train.py --attn soft --train auglistener --selfTrain --aug tasks/R2R/data/aug_paths.json --speaker snap/speaker/state_dict/best_val_unseen_bleu --load previous_btbert_agent/temp/best_val_unseen --pretrain_model_name ./pretrained_hug_models/dicadd/checkpoint-12864 --angleFeatSize 128 --accumulateGrad --featdropout 0.4 --feedback sample --subout max --optim rms --lr 0.00002 --iters 100000 --maxAction 35 --encoderType Dic --batchSize 20 --include_vision True --use_dropout_vision True --d_enc_hidden_size 1024 --critic_dim 1024 --name cvpr_agent
```



You can also start fine-tuning based on previous snapshot by following command. Based on our observation, continue training on previous snapshot and reduce learning rate correspondingly would be helpful.  

```
CUDA_VISIBLE_DEVICES=0 python r2r_src/train.py --attn soft --train auglistener --selfTrain --aug tasks/R2R/data/aug_paths.json --speaker snap/speaker/state_dict/best_val_unseen_bleu --load previous_btbert_agent/temp/best_val_unseen --pretrain_model_name ./pretrained_hug_models/dicadd/checkpoint-12864 --angleFeatSize 128 --accumulateGrad --featdropout 0.4 --feedback sample --subout max --optim rms --lr 0.000002 --iters 100000 --maxAction 35 --encoderType Dic --batchSize 20 --include_vision True --use_dropout_vision True --d_enc_hidden_size 1024 --critic_dim 1024 --d_update_add_layer True --name finetune_cvpr_agent
```
Note: if you come with the cudnn error: CUDNN_STATUS_EXECUTION_FAILED, uninstall torchvision and reinstall torchvision=0.3.0 , eg:
```
conda uninstall torchvision
conda install torchvision=0.3.0
```

#### Train Agent for NDH  

```
python tasks/NDH/ndhtrain.py --path_type player_path --history all --feedback 'sample' --encoder_type 'vlbert’ --eval_type 'val' --batch_size 5 --pretrain_model_name ./pretrained_hug_models/dicadd/checkpoint-12864 --learning_rate 0.0005 --n_iters 20000 --vl_layers 4 --la_layers 9 
```  







