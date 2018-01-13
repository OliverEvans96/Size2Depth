# Size To Depth
Yiran Wu, Sihao Ying, Lianmin Zheng

## Abstract

We consider the problem of single monocular image
depth estimation. It is a notoriously challenging problem
due to its ill-posedness nature. Previous efforts can be
roughly classified into two families: learning-based method
and interactive method. The former, in which deep convolu-
tional neural network (CNN) is adopted frequently, leads to
considerable results on specific dataset, but perform poorly
on images outside the dataset, which shows its lack of ex-
tensiveness. Besides, plenty of data are needed to train
the model. The latter requires human annotation of depth
which, however, is easily to have large errors.
To overcome these problems, we propose a new perspec-
tive for single monocular image depth estimation problem:
size to depth. Most previous interactive methods try to ob-
tain depth labels directly from human. Different from these
methods, our method receives object size labels from human
as prior. Depth can be inferred through simple geometric
relationships given size labels. Then we design a condi-
tional random field (CRF) model to propagate depth infor-
mation and finally generate the whole depth map. We exper-
imentally demonstrate that our method outperforms tradi-
tional depth-labeling methods and can produce satisfactory
depth maps.

## Demo
![video](src)

![demo](res/demo.png)

## Author
<table style="width:100%" align="center" border="0">
  <tr style="border-style:hidden">
     <th style="border-style:hidden"><img src="./res/wyr.jpg" style="border-radius:50%;border-style: solid;border-width: thin;"></th>
     <th style="border-style:hidden"><img src="./res/ysh.jpg" style="border-radius:50%;border-style: solid;border-width: thin;"></th> 
     <th style="border-style:hidden"><img src="./res/zlm.jpg" style="border-radius:50%;border-style: solid;border-width: thin;"></th>
  </tr>
  <tr style="border-style:hidden">
     <th style="border-style:hidden"> <a href="https://github.com/Yiranwu">Yiran Wu</a></th>
     <th style="border-style:hidden"> <a href="https://github.com/yingsihao">Sihao Ying</a></th>
     <th style="border-style:hidden"> <a href="https://github.com/merrymercy">Lianmin Zheng</a></th>
  </tr>
</table>

## Code & Paper
View code on [Github](.).  
View paper on [Here](src).  
