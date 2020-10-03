**LDSCondensenet**

This is a implementation combining condensenet https://github.com/ShichenLiu/CondenseNet with depthwise seperable convolution to gain a more efficient model.

1. We have achieved 74% testing accuracy here which is very good and comparable to other traditional models trained on the Cifar10 dataset.
2. The model name is **Final_model.tar** present in directory root.
3. I have included all the necessary dependencies in the requirements.txt file.
To run this example: install the requirements using `pip -r requirements.txt` in an envoirnment of your choice.

4. Have a look at the load_and_test.ipynb inside jupyter notebook.
    - A Model basic example of loading the model and getting inference is provided there.
    - To run in on your machine open the notebook inside jupyter notebook and  run all cells.
    - The notebook may also be uploaded to google colab and tested there , make sure to include the model file as well.

5. The model is also implemented in the run_model function inside run model.
    - Calling that function by passing in a image will return you the class of the object

Changes: Rewrote the Depth seperable layer due to lack of compatiblity caused by shape problems caused by including the layer in LDSConvnet

Doing this gave us a model with FLOPs: 58.71M, Params: 0.44M  along with a very tiny model size of just 1.78mb.

6. To reproduce the experiment type in `main.py --model condensenet -b 96 -j 16 cifar10 --epochs 300 --lr-type cosine --stages 14-14-14 --growth 8-16-32 --bottleneck 4 --group-1x1 4 --group-3x3 4 --condense-factor 4 --gpu 0 --resume`

7. If running pycharm you may directly run it as the run configurations are included there as well
I hope you had a good experience. 

Regards Parag Chaudhari
