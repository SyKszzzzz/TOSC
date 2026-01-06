CKPT=$1

if [ -z ${CKPT} ]
then
    echo "No ckpt input."
    exit
fi


python sample.py hydra/job_logging=none hydra/hydra_logging=none \
            exp_dir=${CKPT} \
            diffuser=flowgrasp \
            diffuser.loss_type=l1 \
            diffuser.steps=100 \
            model=unet_grasp \
            task=PPT \
            task.dataset.normalize_x=true \
            task.dataset.normalize_x_trans=true
