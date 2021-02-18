python train_ssd.py --dataset /media/2tb/Hoang/multitask/data/CMU_data_origin \
                    --net mb1-ssd --base_net models/mobilenet_v1_with_relu_69_5.pth \
                    --batch_size 32 \
                    --num_epochs 3 \
                    --scheduler cosine \
                    --lr 0.01 \
                    --t_max 200 \
                    --num_workers=4 \
		    --validation_epochs=1
