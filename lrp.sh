# first generate caption.json
CUDA_VISIBLE_DEVICES=0 python generate_caption.py \
  --mode difnet_lrp \
  --exp_name DIFNet_lrp

# generate lrp_result.pkl
CUDA_VISIBLE_DEVICES=0 python lrp_total.py \
  --mode difnet_lrp \
  --exp_name DIFNet_lrp

# show lrp_result
python show_lrp.py --exp_name DIFNet_lrp