# Teacher-student training
- docker: liry_tf
`/raid/liry/tf-kaldi-speaker/egs/voxceleb = /liry_tf/tf-kaldi-speaker/egs/voxceleb`
- train: stage = 7
- evaluation: stage = 9
## Extract speaker posteriors(softmax) and embeddings from the teacher model
Dual-layer knowledge distillation transfers the knowledge from the embedding layer and the logit layer of the teacher model. It considers using the speaker posteriors and speaker embeddings extracted from the teacher model. So we first need to extract speaker posteriors and embeddings from the teacher model. 
- 工作路径：/liry_tf/tf-kaldi-speaker/egs/voxceleb/multitask
- 脚本： extract_softmax_eftdnn.sh stage=8
### 1. extract speaker embedding -> 生成 xvector.scp
`nnet/run_extract_embeddings.sh`
the same as what we did in normal xvector system
### 2. extract speaker posterior 的实现说明 -> 生成 slabel.scp
`nnet/run_extract_softmax_eftdnn.sh`
- 修改trainer：
 `../../../model/trainer_softmax_ftdnn.py extract_asoftmax`
```
from model.loss_extract import extract_asoftmax

features, endpoints = self.entire_network(self.pred_features, self.params, is_training, reuse_variables)
endpoints_loss = extract_asoftmax(features, num_speakers, self.params, is_training, reuse_variables)
self.embeddings = endpoints_loss[self.params.embedding_node]
```

- extract_asoftmax 定义于

`../../../model/loss_extract.py`
```
     with tf.variable_scope(name, reuse=reuse_variables):
         # There are 1 variables in angular softmax: the weight matrix w.
         # The name of w is selected as the same with softmax to enable fine-tuning.
         print(weight_l2_regularizer)
         w = tf.get_variable("output/kernel", [shape_list(features)[1], num_outputs], dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer(),
                             regularizer=tf.contrib.layers.l2_regularizer(weight_l2_regularizer))
         params.dict["softmax_w"] = w
         w_norm = tf.nn.l2_normalize(w, dim=0)

         # If ||x|| is scaled, ||x|| = s
         # Then the logits is s*cos(theta), else ||x||cos(theta)
         logits = tf.matmul(features, w_norm)
         endpoints["logits"] = logits
```

- 提取speaker posterior 需要知道speaker label的个数，可以通过查看教师模型的网络结构得到
- show the architecture of model
/liry_tf/tf-kaldi-speaker/egs/voxceleb/multitask/show.py          
```
 #!/usr/bin/python
 import tensorflow as tf
 import numpy as np
 #import matplotlib.pyplot as plt
 Reader = tf.train.NewCheckpointReader('/liry_tf/tf-kaldi-speaker/egs/voxceleb/v1/exp/xvector_nnet_tdnn_asoftmax_m4_linear_bn_1e-2/nnet/model-90000')
 all_variables = Reader.get_variable_to_shape_map()
 for i in all_variables:
     print(i, all_variables[i])
```
其中tdnn是模型的 name scope
```
('tdnn/tdnn5_bn/moving_variance', [1500])
('tdnn/tdnn3_conv/bias', [64])
('tdnn/tdnn6_bn/moving_mean', [512])
('tdnn/tdnn1_bn/beta', [64])
('tdnn/tdnn1_conv/bias', [64])
('tdnn/tdnn4_dense/bias', [512])
('tdnn/tdnn1_bn/moving_variance', [64])
('tdnn/tdnn5_bn/beta', [1500])
('tdnn/tdnn3_bn/moving_variance', [64])
('tdnn/tdnn2_bn/moving_variance', [64])
('tdnn/tdnn7_bn/gamma', [64])
('tdnn/tdnn5_bn/gamma', [1500])
('tdnn/tdnn6_bn/gamma', [512])
('tdnn/tdnn5_dense/bias', [1500])
('tdnn/tdnn6_bn/beta', [512])
('tdnn/tdnn7_bn/moving_variance', [64])
('tdnn/tdnn6_bn/moving_variance', [512])
('tdnn/tdnn7_bn/beta', [64])
('tdnn/tdnn4_bn/beta', [512])
('tdnn/tdnn6_dense/kernel', [3000, 512])
('tdnn/tdnn2_conv/bias', [64])
('tdnn/tdnn4_bn/moving_mean', [512])
('tdnn/tdnn3_conv/kernel', [1, 7, 64, 64])
('tdnn/tdnn2_bn/gamma', [64])
('tdnn/tdnn4_bn/moving_variance', [512])
('tdnn/tdnn5_dense/kernel', [512, 1500])
('tdnn/tdnn2_bn/moving_mean', [64])
('tdnn/tdnn2_bn/beta', [64])
('tdnn/tdnn1_bn/gamma', [64])
('tdnn/tdnn1_conv/kernel', [1, 5, 30, 64])
('tdnn/tdnn3_bn/beta', [64])
('tdnn/tdnn1_bn/moving_mean', [64])
('tdnn/tdnn7_bn/moving_mean', [64])
('softmax/output/kernel', [64, 7328])
('tdnn/tdnn6_dense/bias', [512])
('tdnn/tdnn4_bn/gamma', [512])
('tdnn/tdnn7_dense/kernel', [512, 64])
('tdnn/tdnn3_bn/gamma', [64])
('tdnn/tdnn3_bn/moving_mean', [64])
('tdnn/tdnn2_conv/kernel', [1, 5, 64, 64])
('tdnn/tdnn5_bn/moving_mean', [1500])
('tdnn/tdnn7_dense/bias', [64])
('tdnn/tdnn4_dense/kernel', [64, 512])
```
- 修改网络模型的输出节点数

`nnet/lib/extract_softmax_eftdnn.py `
```
trainer.build("predict",
                   dim=dim,
                   loss_type="extract_asoftmax",
                   num_speakers=7363)
```

### Create data set for extraction:
- data/voxceleb_train_combined_no_sil/train2
`feats.scp  slabel.scp  spk2utt  spklist  utt2num_frames  utt2spk  xvector.scp`
    - slabel.scp
`ID0190_0163 exp_teacher_softmax/xvectors_voxceleb_train/xvector.1.ark:12`
    - xvector.scp
`ID0190_0143 exp/xvector_nnet_tdnn_asoftmax_m4_linear_bn_1e-2/xvectors_voxceleb_train_combined/xvector.1.ark:18642`
- 学生模型训练过程中用到的所有训练数据都需要提取出speaker posterior 和 embedding
- 训练集要有：feats.scp, slabel.scp, xvector.scp
    - slabel.scp : scp file for speaker posteriors
    - xvector.scp : scp file for speaker emebddings

从 data/voxceleb_combined 筛选出data/voxceleb_train_combined_no_sil/train2 并且 apply cmn -> data/voxceleb_train_extract

`cp -r data/voxceleb_combined data/voxceleb_train_extract`
`cp data/voxceleb_train_combined_no_sil/train2/utt2spk  data/voxceleb_train_extract/.`
`utils/fix_data_dir.sh data/voxceleb_train_extract`
检查路径是否给对
### Define the architecture of student model
- `../../../model/trainer_multitask.py`

` from model.tdnn_ori import tdnn`

### Multi-task training
#### 1. 训练
- 脚本：
`/raid/liry/tf-kaldi-speaker/egs/voxceleb/multitask/run_demo_train.sh`

- 调用训练脚本进行网络训练的命令
` nnet/run_train_nnet_multitask.sh --cmd run.pl --env tf_2_gpu --continue-      training false nnet_conf/tdnn_multitask_DLKD.json /liry_tf/tf-kaldi-speaker/  egs/voxceleb/xatx/data/voxceleb_train_combined_no_sil/train2 /liry_tf/tf-kaldi-speaker/egs/voxceleb/multitask/data/voxceleb_train_combined_no_sil/train2/   spklist /liry_tf/tf-kaldi-speaker/egs/voxceleb/multitask/data/                     voxceleb_train_combined_no_sil/valid2 /liry_tf/tf-kaldi-speaker/egs/voxceleb/ multitask/data/voxceleb_train_combined_no_sil/train2/spklist /liry_tf/tf-kaldi-    speaker/egs/voxceleb/multitask/exp_DLKD_64/DLKD/                                   xvector_nnet_tdnn_asoftmax_m4_linear_bn_1e-2`
我们准备好logit-label and embedding-label 之后就可以进行知识蒸馏

#### 2. 联合训练的代码：model.trainer_multitask
`../../../model/trainer_multitask.py`

- alpha is for embedding-layer distillation
beta is for logit-layer distillation
代码实现包括：
- set value of alpha and beta
- joint-training part
- add place holder for xvector and label
- 可以修改alpha, beta递增的曲线
    `new_alpha = 1-0.99**epoch`
 
```
         graph = tf.get_default_graph()
         le_alpha = graph.get_tensor_by_name('level_alpha:0')
         new_alpha = 1-0.99**epoch
         le_beta = graph.get_tensor_by_name('level_beta:0')
         new_beta = 1-0.99**epoch
         print("*****", self.sess.run(le_alpha))
         print("*****", self.sess.run(le_beta))
         print("*****", new_alpha)
         print("*****", new_beta)
         if new_alpha > self.params.alpha:
             new_alpha = self.params.alpha
         if new_beta > self.params.beta:
             new_beta = self.params.beta
         self.sess.run(tf.assign(le_alpha, new_alpha))
         self.sess.run(tf.assign(le_beta, new_beta))
```

   - build part: joint training
    
```
             embed_distill_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=tdnn6, labels=self.train_xvectors)
             embed_distill_loss_sum = tf.reduce_mean(embed_distill_loss)

             label_distill_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.train_slabel)
             label_distill_loss_sum = tf.reduce_mean(label_distill_loss)
             endpoints.update(endpoints_loss)
             regularization_loss = tf.losses.get_regularization_loss()
             #alpha = tf.Variable(0.0, tf.float32, name='level_alpha')
             alpha = tf.get_variable('level_alpha', shape=(), dtype=tf.float32, initializer=tf.constant_initializer(0),               trainable=False)
             beta = tf.get_variable('level_beta', shape=(), dtype=tf.float32, initializer=tf.constant_initializer(0),                 trainable=False)

             #alpha = tf.Variable(0.00001, tf.float32, name='level_alpha')
             print(alpha.name)
             print(beta.name)

             total_loss = loss + regularization_loss + alpha * embed_distill_loss_sum + beta * label_distill_loss_sum
```


#### 3. 如果修改代码，有一些细节需要注意
- change the dim of label
```
         self.train_xvectors = tf.placeholder(tf.float32, shape=[None, 1024], name="train_xvectors")
         self.train_slabel = tf.placeholder(tf.float32, shape=[None, 7363], name="train_slabel")
```
- spk的数量和训练数据相同
```
data/voxceleb_train_combined_no_sil/train2/spklist
```
- 保证slabel.scp在cat后拼接的时候NF=2


#### 4. 模型配置
- embedding dim =512
- teacher parameter
`30*5*512+512*3*512+512*3*512+512*512+512*1500+3000*512+512*512+512*40`
- student64 parameter
`30*5*64+64*3*64+64*3*64+64*3*64+64*512+512*1500+3000*512`

- student32 parameter
`30*5*32+32*3*32+32*3*32+32*3*32+32*3*32+32*218+218*1000+2000*512`
