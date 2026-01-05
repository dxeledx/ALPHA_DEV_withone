# Motor imagery electroencephalography channel selection based on deep learning: A shallow convolutional neural network

![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-03/cfc8bd27-abc7-421f-97b9-8b4aff104e5d/be2085fa8a49ce5d67d756a9f179dcc26dcdddab5685c0282dc490d7f7d6c56f.jpg)


Homa Kashefi Amiri a, Masoud Zarei b, Mohammad Reza Daliri b,*

<sup>a</sup> Artificial Intelligence Department, School of Computer Engineering, Iran University of Science and Technology (IUST), Narmak, 16846-13114, Tehran, Iran

$^{b}$  School of Electrical Engineering, Iran University of Science and Technology (IUST), Narmak, 16846-13114, Tehran, Iran

# ARTICLEINFO

Keywords:

Electroencephalography

Brain-computer interface

Motor imagery

Deep learning

Convolutional neural network

Pointwise convolution

# ABSTRACT

Electroencephalography (EEG) motor imagery (MI) signals have recently attracted much attention because of their potential to communicate with the surrounding environment in a specific way without the need for muscular and physical movement. Despite these advantages, these signals are difficult to detect due to their low signal to noise (SNR) rate and non-stationary and dynamic nature. Convolutional neural networks (CNN) can extract appropriate spatial and temporal features without the need for separate feature extraction and classification steps. Clean input data for CNN significantly improves its performance, but sometimes, common interference occurs between multi-channel signals. This challenge, along with the noisy EEG signals, degrades the performance of CNN networks. This paper presents a channel selection method based on convolutional neural networks to obviate these challenges. The proposed shallow convolutional neural network (SCNN) is designed with temporal convolution and pointwise convolution (using  $1 \times 1$  convolution) to select the best channel with minimal computational load. Following the channel selection step, the multi-layer fusion CNN model was used to classify the signals. Choosing suitable and relevant EEG channels for motor imagery tasks makes it possible to create appropriate inputs for the classification model so that the model can be improved in the end. For High Gamma Dataset and Brian-Computer Interface (BCI) Competition IV-2a, the following accuracies were obtained:  $81.15\%$  and  $72.01\%$ , which are higher than when the other channel selections such as Mutual Information, Sequential Feature Forward Selection (SFFS) and wrapper methods are used.

# 1. Introduction

Brain-computer interfaces provide an alternative communication method between the human brain and external devices in which physical movement is not required (Al-Saegh et al., 2020). One of the most valuable applications of brain-computer interfaces is to help patients with locked-in syndrome (Chaudhary et al., 2017). A BCI system requires appropriate and discriminative input EEG signals for a practical application. The BCI applications are not limited to helping patients with locked-in syndrome. Also, BCI systems play a successful role in other fields, such as gaming (Ahn, 2014), virtual reality (Coogan et al., 2018), neuromarketing (Rocha et al., 2013), smart houses (Lee et al., 2013), and unmanned aerial vehicles (UAVs) (Vijayendra et al., 2018). Various BCI paradigms are introduced, which are different due to the stimulation method (Wolpaw et al., 2012). One of the important of these paradigms is motor imagery, specifically, the EEG signal. Owing to its non-invasive nature and low cost, this signal has attracted much

attention (Novi et al., 2007). Additionally, motor imagery EEG signals have a high temporal resolution (milliseconds) that cannot be seen even in imaging technologies such as computed tomography (CT) or magnetic resonance imaging (MRI) (Amin et al., 2019). Neurological research has shown that when a subject imagines or actually performs actions like hand or foot movement, a similar area in the subject's brain is activated. This area is called sensorimotor cortex (Pfurtscheller et al., 1999).

Traditional machine learning methods process and classify motor imagery EEG signals (Sreeja et al., 2017). These methods are divided into stages: pre-processing (noise removal and channel selection), feature extraction, feature selection, and classification. On the other hand, deep learning networks can provide an end-to-end system that performs all parts of pre-processing, extraction, feature selection, and classification at once (Dose et al., 2018). Deep learning networks perform remarkably well in BCI systems. However, their performance is highly dependent on the quality of the input signal. Since recorded EEG signals have a low SNR ratio and low spatial resolution, it is essential to

perform appropriate pre-processing steps and extract relevant and discriminative features from these signals (Tibrewal et al., 2022).

Meanwhile, by extracting suitable channels with discriminative information, higher accuracy can be achieved, and the model with less computational power will achieve improved performance. Above and more importantly, brain activity varies between individuals, and choosing the optimal channels for each individual helps to reduce the noise and computational costs of high-dimensional data (Xia et al., 2023).

The main objective of this study is to select channels from high-dimensional EEG data in order to obtain subject-relevant appropriate data. The core contributions of the paper are as follows:

- A channel selection method based on convolutional neural networks is proposed. To design this shallow network, temporal and pointwise convolutions are implemented.

- For classification, a CNN fusion network (Amin et al., 2019) is used. Through the proposed channel selection method and applying clean data to the network, techniques such as data augmentation, pre-trained network, and transfer learning are not used to train the CNN network.

- This channel selection method has achieved high accuracy on the HGD dataset and BCI Competition IV-2a.

The main structure of this paper consists of five sections. The first section is the introduction, which provides information about the background and significance of motor imagery-based BCI systems. The second section is related work that discusses recent research on feature extraction and channel selection of EEG signals. The third section is the main part of the article, which explains the proposed channel selection method. The fourth section discusses the experimental setting and shows the experimental results of the proposed method for benchmark datasets. In the fifth section, the conclusion of the research approach and future prospects are presented.

# 2. Related study

Machine learning has many branches. For example, Meta-learning is a subset of machine learning, which is described as "learning to learn." In (Farhadi et al., 2023), the authors have used Neuro-Fuzzy Meta-Learning (NF-ML) for Unsupervised Domain Adaptation (UDA) in real-world situations. Machine learning algorithms have been widely used to classify EEG signals in BCI applications (Cecotti et al., 2011). Deep networks have been used for various EEG tasks. In (Craik et al., 2019), the authors investigated the answers to some questions. For instance, which types of EEG signals are analyzed using deep learning methods, or are the existing deep learning structures appropriate for different tasks in EEG. A study (Mao et al., 2020) utilized deep learning and especially CNN to classify epileptic seizures in EEG. First, they converted EEG signals to time-frequency domain images and then classified them using CNN networks.

For a better classification performance of EEG signals, it is imperative to give appropriate input data to the classifier. Researchers have recently proposed different approaches to select EEG channels from EEG data (Zeeshan Baig et al., 2020). In (Wang et al., 2019), normalized mutual information is used as a channel selection approach to recognize EEG emotions. Other research has investigated filtering techniques for channel selection in motor imagery EEG-based BCIs. In the wrapper method, the testing and training of each candidate are performed with a specific classification method (Liu et al., 2005). Wrapper methods are computationally more expensive than filtering-based channel selection approaches (Alotaiby et al., 2015). According to (Pawan and Dhiman, 2023), the authors used the Pearson Correlation Coefficient to select the best channels for motor imagery EEG-based BCIs. In the work (Mahamune et al., 2022), researchers utilized the standard deviation of wavelet coefficients for channel selection of motor imagery EEG signals.

In (Guttmann-Flury et al., 2022), EEG channels are selected from source localization methods for four popular BCI paradigms: motor imagery, motor execution, steady-state visual evoked potentials, and P300. As described in (Huang et al., 2023), an approach based on tensor decomposition is used for channel selection of motor imagery signals. Another study (Mahamune et al., 2022) proposes a Fisher score calculation method based on OVR-CSP features for BCI channel selection. In (Liu et al., 2023a), channel selection is considered a multi-objective problem model, and a domain knowledge-assisted multi-objective optimization algorithm (DK-MOEA) is introduced to solve the problem. In recent work, researchers combined a sequential search method with a genetic algorithm for channel selection in EEG-based BCIs and called it Deep Genetic Algorithm Fitness Formation (DGAFF) (Ghorbanzadeh et al., 2023). Recent studies have used deep learning methods for EEG signal classification (Liu et al., 2023a; Ghorbanzadeh et al., 2023; Mzurikwao et al., 2019). In (Yuan et al., 2018), the authors concluded if they reduce the number of EEG channels and use deep learning approaches for classification, the average learning time increases by  $58\%$ . Some studies have used deep learning to select EEG channels. In (Mzurikwao et al., 2019), the authors used extracted weights from the CNN-trained model, and with the reduced number of channels, they achieved a model with superior performance. To address the EEG channel selection issue in BCI applications, researchers (Yuan et al., 2018) proposed a model called ChannelAtt, which is an end-to-end deep learning model using a channel-aware attention mechanism. In another work, authors introduced a compact convolutional neural network (CNN) to select a minimum number of channels, and they used three common BCI paradigms: P300 auditory oddball, steady state visually evoked potential, and motor imagery signals. In (Schirrmeister et al., 2017), a recurrent-convolution neural network is used for learning spatiotemporal representations, and a novel Gradient-Class Activation Mapping (Grad-CAM) visualization is applied for channel selection. For EEG seizure detection, a deep learning algorithm using CNN-LSTM was used on various channel configurations, and each configuration was utilized to minimize the spatial information lost. Researchers in (Thodoroff et al., 2016) introduced a model combining LSTM and autoencoder and an attention channel for inter-subject classification.

Investigations (Lotte et al., 2009) indicate that the accuracy of the BCI model in subject-specific cases has a higher accuracy than in subject-independent ones. This shows that optimal EEG channels are not the same for all subjects. Current work considers the problem of subject variability in EEG-based BCI systems and proposes a shallow, simple multi-layer convolutional neural network for motor imagery EEG channel selection. As a channel selection method, temporal convolution and two pointwise convolutions are used along each channel. Then, the accuracy of each channel is compared with other channels, and K's best channels are selected.

The proposed convolutional network and temporal convolution type, as well as the low depth of the network, have made this proposed architecture work better than other architectures. Moreover, the temporal information of each channel is examined separately, and the role of each channel in the correct classification of each class of motor imagery signal is investigated. And each shallow convolution network with fewer parameters checks which channel has more information. Using  $1 \times 1$  convolution has also caused the depth of feature maps to decrease in each layer without changing the size of the feature maps. Additionally, this type of convolution applies non-linear transformations to the extracted feature maps and obtains better features in each layer without imposing a high computational load on the network and increasing the number of parameters.

# 3. Materials and methods

This section describes the datasets used and the proposed approach for motor imagery EEG channel selection and its classification.

# 3.1. Dataset description

In order to assess the efficacy of the proposed model, two benchmark datasets of motor imagery EEG have been employed. A description of these datasets is presented below.

# 3.1.1. BCI competition IV-2a dataset

The BCI Competition IV-2a dataset (Brunner et al., 2008) is one of the benchmark motor imagery datasets. The data were collected from nine healthy subjects using 25 electrodes. Three of the electrodes are electrooculography (EOG) channels that provide more information regarding eye movement, and in most cases, for EEG analysis, these channels are dropped from the original data. So, 22 EEG electrodes are used. The sampling frequency of the data is  $250\mathrm{Hz}$ , and researchers can access band-pass filtered data between  $0.5\mathrm{Hz}$  and  $100\mathrm{Hz}$ . The dataset contains four different imagined movements and motor imagery of other body parts: left hand, right hand, feet, and tongue. Each subject's data is recorded in two sessions on different days. Most studies use the first session for training and the second for testing. Each subject sat in an armchair before the computer screen during the recording. The structure of each trial is that at  $t = 0$ , a fixed cross and short warning alarm appear on a screen with a black color. After passing  $2\mathrm{s}$  ( $t = 2$ ), an arrow with a specified direction is shown on the screen that relates to one of the four imagery classes. The left arrow is for the left-hand class, the right for the right hand, the down arrow for the feet, and the up arrow for the tongue. These arrows with specific directions are displayed on the screen for about  $1.25\mathrm{s}$ . During this time, the subject is asked to imagine the desired class and keep this imagery until the fixed cross disappears from the screen. The total number of trials for each subject is 288, considering a 2-s rest between trials. Each trial timing is shown in Fig. 1.

# 3.1.2. High Gamma Dataset

The High Gamma dataset (Schirrmeister et al., 2017) is a four-class motor imagery dataset: left hand, right hand, feet, and rest. This dataset also has a training set and a test set. The number of trials in the training and test sets are 880 and 160, respectively. This data is recorded from 14 subjects (6 females, 8 males, age  $27.2 \pm 3.6$  (mean  $\pm$  std)) in a controlled environment. To keep the subjects engaged and active during the trials, they were asked to clench their toes, tap their fingers sequentially, and relax. These particular movements were chosen because the subjects should do little proximal muscular activity. This data was collected using 128 channels and a  $500\mathrm{Hz}$  sampling rate. In total, 13 runs were executed, and the class contained 260 trials. Four seconds were allotted to each motor imagery trial, and the time interval between trials was  $3 - 4\mathrm{s}$ .

# 3.2. Proposed shallow CNNs channel selection model

This study's proposed method for channel selection is composed of several simple shallow CNNs, which are applied separately on each channel. Each shallow CNN is trained on each channel, and the feature vector is extracted from each trained CNN and given to two dense layers to specify the class of each channel. Then, according to the accuracy

![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-03/cfc8bd27-abc7-421f-97b9-8b4aff104e5d/cc742ca2015cf52a3dc50b8bb0278943ded01e6938a312415d5b257fbe7e59f1.jpg)



Fig. 1. Timing pattern of BCI CompetitionIV-2a (Musallam et al., 2021).


obtained for each of them, the best K channels are selected. For this purpose, different combinations of channels are examined according to their accuracies. Finally, a CNN fusion network (Amin et al., 2019) is used to determine the MI classes.

Inspired by the EEGNet network (Lawhern et al., 2018), which first applies convolution along time steps and then convolution on all channels, temporal convolution should be considered because the best channels should be selected, and the channels that play a more significant role in determining the class of each signal need to be taken into account.

In this paper, different architectures were tested to design the channel selection convolutional neural network. By investigating recent studies (Lawhern et al., 2018; Thodoroff et al., 2016; An et al., 2014), it was observed that the deep learning networks that were successful in classifying EEG signals all had shallow architectures. Due to this, shallow CNN networks were tested for the proposed channel selection model. Firstly, a convolution layer, a max pooling, and a dense layer were involved. The need to preserve the features of each channel as much as possible is heightened by using CNN for channel selection rather than classification. While the pooling layer has the advantage of reducing the dimensions of the feature map, it ignores some salient information. The other problem related to convolutional neural networks is that the number of feature maps increases with the addition of network depth, which will eventually cause a dramatic increase in the number of network parameters. Pointwise convolution was employed to preserve the maximum information of each channel and prevent increasing network parameters; this is what was first introduced by (Lin et al., 2013). That study proposed an MLP convolutional layer and used cross-channel pooling to promote learning across channels. To this aim, they introduced a  $1 \times 1$  convolution layer and described it as a cross-channel parametric pooling implementation. It is mentioned in the article that each  $1 \times 1$  convolution kernel, which is equivalent to a channel parametric pooling layer, applies weighted linear recombination on the input. Simply, a  $1 \times 1$  convolution layer is often used as channel-wise pooling to manage the CNN model's complexity. Researchers in (Szegedy et al., 2014) explicitly stated that they used  $1 \times 1$  convolution to reduce dimensions, and in the design of the inception module in the GoogLeNet model, they used  $1 \times 1$  convolution to increase the number of feature maps after pooling. Paper (He et al., 2015) introduced  $1 \times 1$  convolution as a projection method to adapt the number of input filters and output filters.

As it is clear from the above-mentioned studies and the main goal of  $1 \times 1$  convolution, it can have several important effects:

1) Decreasing the depth in each layer without changing the size of the feature map

2) Increasing the depth in each layer without changing the size of the feature map

3) Keeping the same depth in each layer without changing the size of the feature map

4) Applying linear transformations and extracting better features without high computational load or increasing the network parameters

The initial step begins by considering the first and fourth parts, reducing the depth of the temporal features extracted from each channel, and applying linear transformations. As shown in Fig. 2, to reduce the depth of temporal feature maps extracted from each EEG channel, it is enough to apply  $1 \times 1$  convolution with the desired number of filters on the temporal feature maps.

The overall architecture of the proposed CNN-based channel selection network is shown in Fig. 3. First, the signal is band-pass filtered with a frequency band of  $7 - 40\mathrm{Hz}$ . Also, frequency bands, such as  $7 - 13\mathrm{Hz}$ ,  $13 - 40\mathrm{Hz}$ , and  $0 - 40\mathrm{Hz}$ , were examined, and the selected  $7 - 40\mathrm{Hz}$  band had better results than the other bands. In addition, as another preprocessing step of channel selection, signals were standardized with

![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-03/cfc8bd27-abc7-421f-97b9-8b4aff104e5d/9f64f6572b9e0bd4fcce13583cc65047db9b0368a567d3efeb2eaaf554e0d922.jpg)



Fig. 2. The effect of  $1 \times 1$  convolution on reducing feature map dimensions.


mean zero and variance one. As illustrated in Fig. 3, each channel is separated from the original signal and given as an input to channel selection CNN. The architecture of this network is simple and shallow. Its first layer is a temporal convolution layer by the size of  $30 \times 1$ . After that, there are two  $1 \times 1$  convolutions to reduce the depth of feature maps and apply linear transformations. Then, two dense layers are added to the architecture for classification, and the number of neurons in the last layer is equal to the number of classes in the datasets. Finally, the accuracy of each channel is obtained for each network, and according to the accuracy obtained for each channel, the K best channels and their best combinations are determined. Sometimes, a feature may not be good on its own, but combined with other features, it will lead to better performance of the whole model. If each channel is interpreted as a feature, different combinations of these channels can be considered as the network's input. For example, multiple channels alone may not lead to effective model performance, but adding more channels can improve model performance. Therefore, different combinations of the best channels were investigated to select the best one.

After computing the accuracy of each channel with the proposed SCNN network, the channels were sorted in descending order based on the accuracy metric. Then, by choosing a small  $K$  at the beginning (initial  $K = 3$ ) and then increasing it and checking the performance of the Fusion CNN model, the best value of  $K$  was selected. In other words, first  $K = 3$ , the best channels (channels that had the highest accuracy among other channels) were selected, and the Fusion CNN model accuracy was calculated using these three channels. Then,  $K$  was increased in each step, and the accuracy of the Fusion CNN model was calculated. This procedure was continued until there was no further improvement in the Fusion CNN model's performance. The best  $K$  value was 6.

Investigations were conducted into other architectures during our tests for the channel selection model. For instance, two-dimensional convolutions with a horizontal filter like the first layer were utilized instead of  $1 \times 1$  convolutions in the model. However, the proposed model outperformed them due to the fact that in this model,  $1 \times 1$  convolution ensures that not much information is lost in the layers while simultaneously reducing the depth of the feature maps. Networks with higher depth were also examined. However, according to prior studies, the CNN networks that performed well for EEG all had low depth, and networks with higher depth resulted in degraded performance. Other deep networks, such as LSTM, were also examined, which can be used to examine and classify the information of each channel. However, the RNN family models performed worse than CNNs.

After pre-processing and channel selection, we should classify motor imagery EEG signals. In recent studies, EEG signals have been applied in different ways as input to convolutional neural networks. Since CNN networks were first developed for two-dimensional images, some researchers first converted EEG signals into topo-maps and images and then applied them to CNN networks (Thodoroff et al., 2016). By converting the EEG signals to images, some important information may be lost, and it is only possible to preserve some generic information. Some other works (Sadiq et al., 2022) have first extracted features such as wavelets from EEG signals, and in the next step, these features have been

![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-03/cfc8bd27-abc7-421f-97b9-8b4aff104e5d/c126756748cc87ea8b9fcaac9b2325be979edd50e358dfb4127d5cd9fc22df75.jpg)



Fig. 3. The Proposed CNN-based channel selection model. After separating each channel from the trial, firstly, 50 temporal filters with the size of 30 by 1 are applied to the signal, and then 30 and 20 pointwise or  $1 \times 1$  convolution are applied and obtained feature maps are added to the dense layer with 1024 neurons. Finally, a dense layer with four neurons is added so that the four-class classification is done for each channel. Based on the accuracy obtained, the combination of the best channels is selected.


given to the CNN network as features. Although this approach can perform well in the classification of EEG signals, more is needed to provide the main goal of deep learning networks, which is the development of an end-to-end system. The third method used in the studies is applying the EEG signal as a two-dimensional input of time samples across channels (Schirrmeister et al., 2017). In order to classify, the third approach is used; signals are prepared in the form of two-dimensional matrices and are used in multi-layer fusion CNN (Amin et al., 2019). This CNN-based model, which is proposed for motor imagery EEG signal classification, achieved high accuracy on two datasets, the High Gamma dataset (He et al., 2015) and BCI Competition-IV 2a (Lin et al., 2013).

The overall architecture of this network is shown in Fig. 4. This network is a four-layer fusion CNN. Firstly, each trial of the motor imagery EEG signal with four different bands of  $7 - 13\mathrm{Hz}$ ,  $13 - 31\mathrm{Hz}$ ,  $7 - 31\mathrm{Hz}$ , and  $0 - 40\mathrm{Hz}$  are filtered using a band-pass Butterworth filter. Then, each filtered signal enters a CNN network and four CNNs process four filtered EEG signals with different bands. Afterward, the fusion CNN model extracts the obtained features (the layer before Softmax) from all four CNN networks, concatenates them together, and applies this concatenated feature to a two-layer MLP and trains it. The output of this MLP is given to a Softmax layer, and the accuracy of each trial will be obtained.

The training conditions of this network are similar to the original article (Amin et al., 2019) except for two cases: first, no pre-training was used to train this fusion CNN network, and secondly, the data was not increased, or in other words, data augmentation was not used. Since the number of channels or classes in motor imagery EEG datasets is different, pre-training sometimes becomes difficult, and it is necessary to change the network settings. As a result, the network's weights obtained in the pre-training stage couldn't be suitable and practical for the test stage. On the other hand, since most deep networks require a large amount of data for training and motor imagery, EEG datasets usually have a small number of trials; most of the previous articles have tried to increase this small number of trials using different methods, such as sliding windows. In this way, many trials are made from each trial to provide the data needed for training deep networks. However, the problem with this approach is that the generated data are generally similar and do not lead to better network learning. Therefore, increasing

the number of data to train the network will only add redundancy and computational load to it and will have little effect on network training. However, methods that generate meaningful synthetic data can lead to better network performance. In this article, CNN fusion networks are trained without using a pre-training approach and data augmentation, so the computational load is less and the training time is shorter, and this is the difference and advantage of this work compared to previous similar works.

Some limitations came up when the model architecture was being designed and trained. Initially, a suitable system for experiments was required. The models were trained using the Google Colab environment. However, the free Google Colab subscription alone was not sufficient for training, and for this reason, Google Colab Pro and Google Colab Pro + subscriptions were utilized. The model, which was composed of two deep networks - the first for channel selection and the second for classification, resulted in high training times for the two challenging datasets, HGD and BCICIV-2a. Another limitation, which can be considered a limitation of all deep learning methods, is the time-consuming nature of examining the sample models in terms of both time and computing resources. Hence, similar to other machine learning models, evaluating every possible model might not be practical, and it is advisable to economize on experiments.

# 4. Training the CNN-based channel selection method

To train the CNN-based channel selection network, first, each trial was band-pass filtered with a frequency band of  $7 - 40\mathrm{Hz}$ . Standardization with mean and variance was applied for continuous EEG signals. The 3-s EEG signals were considered to be fed to the CNN network. In these cropped 3-s signals, the motor imagery task is done by subjects. Signal cropping using the sliding window approach has been used in many types of research related to EEG signal processing and classification using CNN networks. As mentioned, we did not use the sliding window approach to increase the network's training data. Despite testing this method, by increasing the EEG data using cropping, the performance of the CNN network did not improve significantly. For this purpose, the number of data is increased up to tens of times to measure the effect of this process on the overall performance of the network. It

![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-03/cfc8bd27-abc7-421f-97b9-8b4aff104e5d/89d1879c30dcc0810e4c17b36cac7dbbcc55c7dcc237a536a59ab7021a3ace81.jpg)



Fig. 4. Multi-layer fusion CNN (Amin et al., 2019) as a classification model. The size and the number of filters in each layer are illustrated.


realized that its main reason is that augmented data only helps improve the performance of CNNs when these data are diverse and learn the network new information about the context. Given the above reasons, the data from the main trials for training the proposed CNN network is used.

In the investigation of the performance of the CNN-based channel selection model with different CNN network architectures, the first layer of convolution, which is used to process EEG signals, is divided into two parts: The first convolution performed across time-samples is a temporal convolution and the second convolution which is done across all the channels and is a spatial convolution. Since the purpose of the proposed CNN network is channel selection, and each channel is supposed to be given to the network separately, only the first temporal convolution was considered. Then, after the convolution layer, max pooling and dense layer are put with the Softmax activation function. Also, different architectures with two layers and three layers or more convolutional layers and max pooling layers between them were examined. Although the max pooling layer reduces the size of the feature map, it also destroys its essential information. This information is needed to design the channel selection system. Subsequently, various architectures were investigated, and a  $1 \times 1$  convolutional filter instead of max pooling resulted in better model performance. Different model architectures with temporal convolution layer and  $1 \times 1$  convolution were tested. Finally, the architecture shown in Fig. 5 for each channel resulted in the best performance for the channel selection model.

Table 1 provides a comprehensive depiction of the DNN model parameters. A Batch Normalization (BatchNorm) layer is incorporated subsequent to every convolutional layer to accelerate the training process. Exponential Linear Unit (ELU) activation function is applied following each BatchNorm layer. L2 regularization is employed, and a dropout rate of 0.5 is introduced when the initial fully connected layer is reached to mitigate the issue of overfitting.

The optimal network architecture was determined by trial-and-error method. The number and size of filters were tested according to previous studies and experimental methods, and finally, the best ones were

![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-03/cfc8bd27-abc7-421f-97b9-8b4aff104e5d/0f6c18213014a0a4e6d87cf96a02f81ac49fa999a095b5fa1f4997c6db63fe2b.jpg)



Fig. 5. The Proposed CNN Architecture for each channel.



Table 1 Model Summary of the proposed SCNN.


<table><tr><td>No</td><td>Layer Name</td><td>Layer parameters</td><td>Output shape</td><td>Number of parameters</td></tr><tr><td>1</td><td>2DConvolution</td><td>Filters = 50, kernel_size = (30,1), input_shape = (1001,1), padding = &#x27;same&#x27;</td><td>(1001,50)</td><td>1550</td></tr><tr><td>2</td><td>BatchNorm</td><td></td><td>(1001,50)</td><td>200</td></tr><tr><td>3</td><td>Activation</td><td>ELU</td><td>(1001,50)</td><td>0</td></tr><tr><td>4</td><td>2DConvolution</td><td>Filters = 30, kernel_size = (1,1)</td><td>(1001,30)</td><td>1530</td></tr><tr><td>5</td><td>BatchNorm</td><td></td><td>(1001,30)</td><td>120</td></tr><tr><td>6</td><td>Activation</td><td>ELU</td><td>(1001,30)</td><td>0</td></tr><tr><td>7</td><td>2DConvolution</td><td>Filters = 20, kernel_size = (1,1)</td><td>(1001,20)</td><td>620</td></tr><tr><td>8</td><td>BatchNorm</td><td></td><td>(1001,20)</td><td>80</td></tr><tr><td>9</td><td>Activation</td><td>ELU</td><td>(1001,20)</td><td>0</td></tr><tr><td>10</td><td>Dropout</td><td>Rate = 0.5</td><td>(1001,20)</td><td>0</td></tr><tr><td>11</td><td>Flatten</td><td></td><td>20020</td><td>0</td></tr><tr><td>12</td><td>Dense</td><td>Unit = 1024, activation = &quot;ELU&quot;, kernel.regularizer = L2 (0.01), kernel_initializer = &#x27;uniform&#x27;</td><td>1024</td><td>20501504</td></tr><tr><td>13</td><td>Dense</td><td>Unit = 4, activation = &quot;softmax&quot;</td><td>4</td><td>4100</td></tr><tr><td>Total</td><td></td><td></td><td></td><td>20509704</td></tr></table>

selected according to Fig. 5. Various activation functions were investigated, and eventually, ELU (Exponential Linear Units) was selected as the activation function. Between the ELU and ReLU activation functions, the ELU activation function was faster and led to better network performance. ELU is an activation function that helps to converge cost to zero faster, and its produced results are more accurate. Unlike other activation functions, the ELU activation function also has an alpha constant, which must be set to a positive value:

$$
\boldsymbol {f} (\mathbf {z}) = \left\{ \begin{array}{c c} \mathbf {z} & \mathbf {z} > \mathbf {0} \\ \boldsymbol {\alpha}. \left(\boldsymbol {e} ^ {\mathbf {z}} - \mathbf {1}\right) & \mathbf {z} \leq \mathbf {0} \end{array} \right\} \tag {1}
$$

After convolutional layers, batch normalization (BN) is applied (Loffe et al., 2015). Its main purpose is to normalize the input of the next layer, and through this, both the training speed increases and it will have a regularization effect. BN normalizes the entire batch and zero-centers it. For normalization, it should estimate the mean  $\mu$  and variance  $\sigma^2$  computed in a batch:

$$
\mu = \frac {1}{b} \sum_ {i = 1} ^ {b} X ^ {(i)} \tag {2}
$$

$$
\sigma^ {2} = \frac {1}{b} \sum_ {i = 1} ^ {b} \left(\boldsymbol {X} ^ {(i)} - \mu\right) ^ {2} \tag {3}
$$

In which  $b$  is the total number of training instances in the batch and  $X^{(i)}$  indicates a training instance. To calculate the zero-centered normalized value of  $X^{(i)}$ , equation (4) is used.

$$
\widehat {\boldsymbol {X}} ^ {(i)} = \frac {\boldsymbol {X} ^ {(i)} - \boldsymbol {\mu}}{\sqrt {\sigma^ {2} + \xi}} \boldsymbol {\xi} = \mathbf {1 0} ^ {- 5} \tag {4}
$$

For scaling and offsetting values, equation (5) is used.

$$
\boldsymbol {z} ^ {i} = \gamma \bigotimes \widehat {\boldsymbol {X}} ^ {(i)} + \boldsymbol {\beta} \tag {5}
$$

Where  $\gamma$  is the corresponding scaling parameter, and  $\beta$  is the offset parameter (both of these parameters are trained through backpropagation). Operator  $\otimes$  indicates element-wise multiplication.

As shown in Fig. 4, the model's architecture also uses Dropout layers. This layer helps to improve model performance and prevent model

overfitting. The main idea of Dropout is to drop some nodes and their connected weights randomly during training (Srivastava et al., 2014). The Dropout approach prevents too high co-adaptation of nodes.

The last output layer with a Softmax activation function is used to obtain the probabilities  $\widehat{y}_i$  of four class Motor imagery EEG data:

$$
\widehat {y} _ {i} = \operatorname {a r g m a x} \left(\frac {e ^ {y _ {i}}}{\sum_ {i = 1} ^ {5} e ^ {y _ {i}}}\right) \tag {6}
$$

Adam algorithm is used as an optimization method to optimize model weights and biases. Adam is an effective and suitable optimizer for high-dimensional parameters compared to other optimizers. It is a mini-batch stochastic gradient optimization method. This optimizer was used with a learning rate  $= 10^{4}$ . Various learning rates were tested, and this value performs better than others. The optimization problem of this CNN network's parameters was solved using the categorical cross-entropy loss function (Dai et al., 2020):

$$
\boldsymbol {L o s s} = - \sum_ {i = 1} ^ {4} \left(\mathbf {y} _ {i} ^ {*} \cdot \log \hat {\mathbf {y}} _ {i}\right) \tag {7}
$$

In the training process, a constant number of epochs is not determined. The validation early stopping technique (Prechelt, 2002) is used in which training proceeds until the validation loss does not improve for at least the pre-defined threshold value for a number of consecutive epochs. Additionally, the model checkpoint technique (Siskind et al., 2018) enhances the training efficiency. In this approach, validation loss is checked after each epoch, and if the validation loss is not improved in that epoch, the parameters of the previous epoch will be used for the next epoch. In other words, epochs that perform poorly will be discarded.

The method and conditions of training the multi-layer fusion CNN are exactly the same as in the original paper (Amin et al., 2019). This feature fusion model uses various CNN architectures and extracts domain-specific knowledge and discriminative features that improve the EEG classification accuracy. This architecture is composed of four CNN models: CNN-1, CNN-2, CNN-3 and CNN-4. Each network has a particular depth and extracts features from a specific band. The considered frequency bands for these four networks are  $7 - 13\mathrm{Hz}$ ,  $13 - 31\mathrm{Hz}$ ,  $7 - 31\mathrm{Hz}$ , and  $0 - 40\mathrm{Hz}$ , respectively. In the original paper, two different model architectures (MLP and autoencoder) are used after extracting the features. In the current work, only the MLP model is used as a fusion model. This MLP is constructed from two hidden layers, each having 50 nodes. The features from each CNN are obtained by removing the last Softmax classification layer. A linear layer is used to concatenate the features. Finally, the concatenated feature resulting from multi-layer CNN is fed to the MLP. This concatenated feature vector is used to train the MLP network. The MLP's output is used as an input to a Softmax layer to characterize the probability score for each trial.

In order to train both parts of channel selection and classification, the k-fold cross-validation approach has been used  $(\mathrm{K} = 5)$ . This approach is jointly applied to two sections of channel selection and classification. This means that if an instance is considered training data for the channel selection part, the same instance is considered training data in the classification section. This is done to prove the validity of the results. Following the channel selection process, a thorough evaluation of the K best channels and various channel combinations is performed to determine the perfect channel selections and their combinations of inputs for the classification section. On average,  $K = 6$  in both datasets led to a high model performance. By determining the value of K in the first part of the channel selection, the second layer of the multi-layer fusion CNN model, a convolutional layer with the dimension of  $1\times 22$  is changed to the dimension of  $1\times K$ . This convolutional layer is supposed to be applied across the EEG signal channels and its spatial features extracted; therefore, its dimensions should be proportional to the number of input EEG signal channels.

# 5. Experiments and results

The experiments were performed using Google Colab Pro with TPU backend, and 35 GB RAM and 225 GB Disk RAM were used to train and test deep learning models. Tensorflow framework was used to develop models. Dataset reading and initial preprocessing was done using MNE-Python. MNE is an open-source Python package for EEG and MEG signal processing. This toolbox is written in Python programming language and available from the PyPI package repository.

The accuracy metric was used to evaluate and compare our proposed model with other channel selection models.

$$
A c c u r a c y = \frac {T P + T N}{(T P + T N + F P + F N)} \tag {8}
$$

The accuracy of classification is a commonly used performance metric and is given in equation (8), where  $\mathrm{TP} =$  true positive,  $\mathrm{TN} =$  true negative,  $\mathrm{FP} =$  false positive, and  $\mathrm{FN} =$  false negative.

To evaluate the effectiveness of the proposed channel selection model, we obtained the accuracy of the multi-layer fusion CNN model with and without using the proposed channel selection method for each subject in BCI Competition IV-2a and HGD datasets. The results are shown in Tables 2 and 3. As shown in Tables 1 and 2, the classification accuracy for two datasets has increased by applying the proposed channel selection method. These results indicate that the selection of suitable and informative channels has a positive effect on the classification of CNN performance.

In order to compare the performance of the CNN classification model with and without channel selection, bar plots for two datasets are shown in Fig. 6 and Fig. 7.

As apparent in the above plots and tables, the classification accuracies for both datasets are increased by about  $10\%$  with the proposed channel selection method.

# 5.1. Comparison with other channel selection method

In order to validate the proposed channel selection method, its performance has been compared with three other channel selection methods: Mutual Information, Sequential Feature Selection, and Wrapper Method. The results show that the proposed channel selection method has performed better than the three mentioned methods in both datasets.

# 5.1.1. Mutual information

By considering two discrete variables,  $X$  and  $Y$ , mutual information can be computed using equation (9):

$$
\boldsymbol {I} (\boldsymbol {X}, \boldsymbol {Y}) = \sum_ {\boldsymbol {y} \in \mathcal {Y}} \sum_ {\boldsymbol {x} \in \boldsymbol {X}} \boldsymbol {p} (\boldsymbol {x}, \boldsymbol {y}) \log \left(\frac {\boldsymbol {p} (\boldsymbol {x} , \boldsymbol {y})}{\boldsymbol {p} (\boldsymbol {x}) \boldsymbol {p} (\boldsymbol {y})}\right) \tag {9}
$$

Where  $p(x)$  and  $p(y)$  are marginal probability distribution functions of  $X$  and  $Y$ ,  $p(x,y)$  indicates the joint probability distribution function of  $X$  and  $Y$ . The higher the mutual information value, the more predictive the corresponding feature is for class membership.

# 5.1.2. Sequential feature selection algorithm

The sequential feature selection algorithm searches the channel space iteratively to select the best channels (Pudil et al., 1994). This algorithm starts with an empty set and only adds the channel that provides maximum value for the objective function. Then, the remaining channels are added one by one, and the new channel subset is evaluated. Sequential floating forward selection (SFFS), which is an advanced version of SFS, adds a backtracking step to the main algorithm (Reunanen et al., 2003). Adding a backtracking step eliminates one channel at a time from the subset, and the algorithm evaluates the new channel subset.


Table 2 Classification results with and without proposed channel selection method for BCI Competition IV-2a dataset.


<table><tr><td>Model/Subject</td><td>Sub1</td><td>Sub2</td><td>Sub3</td><td>Sub4</td><td>Sub5</td><td>Sub6</td><td>Sub7</td><td>Sub8</td><td>Sub9</td><td>Average</td></tr><tr><td>Classification accuracy without proposed Channel Selection method</td><td>67.7</td><td>59.55</td><td>68.62</td><td>55.1</td><td>54.26</td><td>40.28</td><td>71.81</td><td>70.74</td><td>65.8</td><td>61.54</td></tr><tr><td>Classification accuracy with proposed Channel Selection method</td><td>76.32</td><td>70.91</td><td>79.12</td><td>67.94</td><td>70.39</td><td>45.31</td><td>85.3</td><td>76.54</td><td>76.31</td><td>72.01</td></tr></table>


Table 3 Classification results with and without proposed channel selection method for High Gamma Dataset.


<table><tr><td>Model/Subject</td><td>Sub1</td><td>Sub2</td><td>Sub3</td><td>Sub4</td><td>Sub5</td><td>Sub6</td><td>Sub7</td><td>Sub8</td><td>Sub9</td><td>Sub10</td><td>Sub11</td><td>Sub12</td><td>Sub13</td><td>Sub14</td><td>Average</td></tr><tr><td>Classification accuracy without proposed Channel Selection method</td><td>62.81</td><td>79.84</td><td>69.37</td><td>73.45</td><td>71.2</td><td>75.63</td><td>77.5</td><td>65.3</td><td>72.8</td><td>78.12</td><td>76.56</td><td>68.75</td><td>77.19</td><td>70.63</td><td>72.79</td></tr><tr><td>Classification accuracy with the proposed Channel Selection method</td><td>74.21</td><td>85.72</td><td>77.69</td><td>77.3</td><td>82.9</td><td>85.32</td><td>82.19</td><td>76.12</td><td>80.63</td><td>87.75</td><td>79.8</td><td>78.5</td><td>86.38</td><td>81.61</td><td>81.15</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-03/cfc8bd27-abc7-421f-97b9-8b4aff104e5d/c81c60963cf557714bcc2b03eeeaac058856ba23d24e98120ec347e077b3fcb2.jpg)



Fig. 6. Bar plot of the effect of the proposed channel selection method on classification accuracy of BCI Competition IV-2a Dataset.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-03/cfc8bd27-abc7-421f-97b9-8b4aff104e5d/75ddacdf2be9c1b459fc78a6725269ab6fe1a3966ed9a5b51877f4b67ce7a6c2.jpg)



Fig. 7. Bar plot of the effect of the proposed channel selection method on classification accuracy of High Gamma Dataset.


# 5.1.3. Wrapper method

A classification algorithm is used in the wrapper methods to evaluate the sample channel subsets, and a classifier is used to generate these channel subsets. To evaluate every candidate channel subset, a specific classifier is trained and tested. This process continues until the best subset is found and a predefined threshold value is reached. Although the wrapper method often performs well, it is computationally expensive and highly prone to overfitting (Liu et al., 2005).

The performance of this proposed model was compared with other conventional channel selection methods described above in order to assess its performance. In these models, instead of the proposed channel selection method, these three channel selection methods are applied to the motor imagery EEG signal, and classification is done using multilayer fusion CNN. In Table 4, the performance of the above three channel selection methods is compared with the proposed method for

each subject of the HGD dataset, and the average values indicate the superiority of our method compared to other methods. In Table 5, the same approach was used for the subjects of the BCI Competition IV-2a dataset, and their average values were compared in the last column. In this dataset, the proposed channel selection method has performed better than other channel selection methods.

For visual comparison, the bar plots shown in Figs. 8 and 9 were provided. It is clear from these two figures that the proposed channel selection method performs better than all three channel selection methods explained in both the High Gamma Dataset and BCI Competition IV-2a.

The proposed model is not only applicable to channel selection in EEG signals but also has the potential to be used for other challenging tasks. For example, in the  $\mathrm{ADC} + +$  (Zhang et al., 2023) model, which is presented for co-salient object detection and is designed with the


Table 4 Comparison of Classification results of the proposed channel selection method with Mutual Information, SFFS, and Wrapper method for High Gamma Dataset.


<table><tr><td>Model/Subject</td><td>Sub1</td><td>Sub2</td><td>Sub3</td><td>Sub4</td><td>Sub5</td><td>Sub6</td><td>Sub7</td><td>Sub8</td><td>Sub9</td><td>Sub10</td><td>Sub11</td><td>Sub12</td><td>Sub13</td><td>Sub14</td><td>Average</td></tr><tr><td>Mutual Information</td><td>67.91</td><td>81.27</td><td>75.31</td><td>75.34</td><td>74.53</td><td>76.92</td><td>78.1</td><td>69.43</td><td>79.43</td><td>80.81</td><td>78.92</td><td>70.78</td><td>82.71</td><td>73.42</td><td>76.06</td></tr><tr><td>SFFS</td><td>68.43</td><td>86.59</td><td>78.44</td><td>74.57</td><td>75.96</td><td>81.94</td><td>79.19</td><td>74.89</td><td>81.45</td><td>83.61</td><td>80.77</td><td>73.6</td><td>84.87</td><td>75.7</td><td>78.57</td></tr><tr><td>Wrapper</td><td>66.87</td><td>87.82</td><td>73.8</td><td>75.5</td><td>79.31</td><td>86.35</td><td>85.46</td><td>70.41</td><td>75.44</td><td>86.66</td><td>78.94</td><td>71.38</td><td>81.69</td><td>77.84</td><td>78.38</td></tr><tr><td>Proposed Channel Selection Method</td><td>74.21</td><td>85.72</td><td>77.69</td><td>77.3</td><td>82.9</td><td>85.32</td><td>82.19</td><td>76.12</td><td>80.63</td><td>87.75</td><td>79.8</td><td>78.5</td><td>86.38</td><td>81.61</td><td>81.15</td></tr></table>


Table 5 Comparison of Classification results of the proposed channel selection method with Mutual Information, SFFS, and Wrapper method for BCI Competition IV-2a.


<table><tr><td>Model/Subject</td><td>Sub1</td><td>Sub2</td><td>Sub3</td><td>Sub4</td><td>Sub5</td><td>Sub6</td><td>Sub7</td><td>Sub8</td><td>Sub9</td><td>Average</td></tr><tr><td>Mutual Information</td><td>70.44</td><td>67.78</td><td>70.28</td><td>59.28</td><td>62.7</td><td>42.5</td><td>73.88</td><td>76.59</td><td>71.37</td><td>66.09</td></tr><tr><td>SFFS</td><td>78.81</td><td>63.67</td><td>77.13</td><td>58.86</td><td>59.44</td><td>46.76</td><td>74.69</td><td>71.41</td><td>69.5</td><td>66.68</td></tr><tr><td>Wrapper</td><td>79.65</td><td>73.4</td><td>75.7</td><td>60.31</td><td>63.91</td><td>48.54</td><td>78.62</td><td>78.7</td><td>69.68</td><td>69.83</td></tr><tr><td>Proposed Channel Selection method</td><td>76.32</td><td>70.91</td><td>79.12</td><td>67.94</td><td>70.39</td><td>45.31</td><td>85.3</td><td>76.54</td><td>76.31</td><td>72.01</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-03/cfc8bd27-abc7-421f-97b9-8b4aff104e5d/4689f6e31cdf099280ab72029b613b1b405e8ddb4efdb30c0af552bb90c2ba94.jpg)



Fig. 8. Comparison of the proposed method with other channel selection methods for the High Gamma Dataset.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-03/cfc8bd27-abc7-421f-97b9-8b4aff104e5d/0f38d6b98d40ffa0f85678662bdb2c3e7de805dcffbd5b1c1f04465434852ec2.jpg)



Fig. 9. Comparison of the proposed method with other channel selection methods for BCI Competition IV-2a.


inspiration of human perspective, it first summarizes the consensus cues from the whole group and then looks for co-salient objects in each image. This proposed model can be used in the  $\mathrm{ADC}++$  model summarization stage to extract the features of each input and then merge them together to produce consensus features. Also, in (Li et al., 2023),

the authors designed a dynamic searching process module for robust perception and precise segmentation to perform multi-modal and multi-scale feature fusion, and then a learning mechanism was proposed to perform robust saliency perceptual learning. In the first part, the proposed model can be used for feature extraction and feature fusion. In

(Liu et al., 2023b), a new ST transformer is introduced for Video Salient Object Detection, which has two branches. In the first branch, the overall context is extracted from two neighbouring frames using the attention mechanism, and in the second branch, long-term temporal information from consecutive frames is fused together. The proposed model of this article has the potential to be employed in the second branch, where temporal information is supposed to be extracted.

# 6. Conclusion

Statistical methods and machine learning are used to select EEG channels in brain-computer interface applications. However, deep learning methods have shown that they perform more effectively than machine learning methods in some cases. With the aim of increasing the performance and accuracy of the classification model, these methods select suitable channels from all the EEG channels. Considering the variety of EEG signals of every subject, these methods try to select the best channels for each subject separately and independently. This paper proposed a CNN-based channel selection network to select the appropriate motor imagery EEG channels for each subject. Compared to other channel selection methods based on CNN, the proposed method has a shallow and simple architecture, and its trainable parameters are less. Despite this, it has shown high performance. To design this architecture, firstly, the signal of each channel was separated and given as an input to a temporal convolution, and then we used two  $1 \times 1$  convolution. We used  $1 \times 1$  convolution to perform linear transformations on the feature map and reduce the depth of the feature map without losing much information. Compared to  $1 \times 1$  convolution, the max pooling operation removes less information in the feature map and thus helps generate more generic features from each channel. Finally, the classification for each channel is done using two dense layers, and the accuracy obtained for each channel is compared with that of other channels. The best channels and their best combination were selected and given as input data to the multi-layer fusion CNN network for classification.

Extensive experiments and evaluations are conducted on challenging datasets to prove the superiority of this CNN-based channel selection method over other methods. For this purpose, three other channel selection methods, Mutual Information, SFFS, and Wrapper method, were used, and the proposed method performed better than all these three channel selection methods in both datasets and all subjects. Since this CNN network can extract generic features from each channel, it can potentially be used for EEG signals other than motor imagery signals. Since the proposed method has achieved this good performance without using transfer learning or increasing the number of input EEG data, our goal in future research is to generate significant artificial data or artificial subjects to increase the model accuracy. Moreover, future studies can improve the proposed channel selection network. For the channel combination section, CNN networks can also be used; the best channels can be selected based on them. Also, other deep neural networks, such as RNN or LSTM, can be used to select suitable EEG channels.

# CRediT authorship contribution statement

Homa Kashefi Amiri: Writing - review & editing, Writing - original draft, Methodology, Formal analysis, Conceptualization. Masoud Zarei: Writing - review & editing, Formal analysis. Mohammad Reza Daliri: Writing - review & editing, Validation, Supervision, Resources, Methodology, Conceptualization.

# Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

# Data availability

The data used in this study is publicly available and the information has been provided in the paper.

# References



Al-Saegh, A., Dawwd, S.A., Abdul-Jabbar, J.M., 2020. Deep learning for motor imagery EEG-based classification: a review. Biomed. Signal Process Control 63, 102172.





Ahn, M., et al., 2014. A review of brain-computer interface games and an opinion survey from researches, developers and users. Sensors 14 (8), 14601-14633.





Alotaiby, T., et al., 2015. A review of channel selection algorithms for EEG signal processing. EURASIP J. Appl. Signal Process 66.





Amin, S.U., et al., 2019. Deep Learning for EEG motor imagery classification based on multi-layer CNNs feature fusion. Future Generat. Comput. Syst. 101, 542-554.





An, X., et al., 2014. A deep learning method for classification of EEG data based on motor imagery. Intell. Comput. Bioinf. 203-210.





Brunner, C., et al., 2008. BCI Competition 2008-Graz Data Set A and B, Institute for Knowledge Discovery (Laboratory of Brain-Computer Interfaces). Graz University of Technology, pp. 136-142.





Cecotti, H., Fraser, A., 2011. Convolutional neural networks for P300 detection with application to brain-computer interfaces. IEEE Trans. Pattern Anal. Mach. Intell. 33 (3).





Chaudhary, U., et al., 2017. Brain-computer interface-based communication in the completely locked-in state. PLoS Biol. 17 (12), e3000607.





Coogan, C.G., He, B., 2018. Brain-computer interface control in a virtual reality environment and applications for the internet of things. IEEE Access 6, 10840-10849.





Craik, A., He, Y., Contreras-Vidal, J.L., 2019. Deep learning for electroencephalogram (EEG) classification tasks: a review. J. Neural. Eng. 16.





Dai, G., et al., 2020. HS-CNN: a CNN with hybrid convolution scale for EEG motor imagery classification. Neural Eng. 17, 016025.





Dose, H., et al., 2018. An end-to-end deep learning approach to MI-EEG signal classification for BCIs. Expert Syst. Appl. 114, 532-542.





Farhadi, A., Sharifi, A., 2023. Leveraging meta-learning to improve unsupervised domain adaptation. Comput. J. 67 (5), 1838-1850.





Ghorbanzadeh, G., et al., 2023. DGAFF: deep genetic algorithm fitness formation for EEG Bio-Signal channel selection. Biomed. Signal Process Control 79 (Part 1), 104119.





Guttmann-Flury, E., Sheng, X., Zhu, X., 2022. Channel selection from source localization: a review of four EEG-based brain-computer interfaces paradigms. Behav. Res. Methods 55, 1980-2003.





He, K., et al., 2015. Deep Residual Learning for Image Recognition, vol. 1. Computer Vision and Pattern Recognition.





Huang, Z., Wei, Q., 2023. Tensor Decomposition-Based Channel Selection for Motor Imagery-Based Brain-Computer Interfaces. Cognitive Neurodynamics.





Lawhern, V.J., et al., 2018. EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. J. Neural. Eng. 15, 056013.





Lee, W.T., Nisar, H., Malik, A., 2013. A brain compuer interface for smart home control. In: 2013 IEEE International Symposium on Consumer Electronics (ISCE).





Li, L., et al., 2023. Robust perception and precise segmentation for scribble-supervised rgb-d saliency detection. IEEE Trans. Pattern Anal. Mach. Intell. 46 (1).





Lin, M., Chen, Q., Yan, S., 2013. Network in network. Neural Evol. Comput. 3.





Liu, H., Yu, L., 2005. Toward integrating feature selection algorithms for classification and clustering. IEEE Trans. Knowl. Data Eng. 17 (4).





Liu, T., Ye, A., 2023a. Domain knowledge-assisted multi-objective evolutionary algorithm for channel selection in brain-computer interface systems. Front. Neurosci. 7 (17), 1251968.





Liu, N., et al., 2023b. Learning complementary spatial-temporal transformer for video salient object detection. IEEE Transact. Neural Networks Learn. Syst. https://doi.org/10.1109/TNNLS.2023.3243246.





Loffe, S., Szegedy, C., 2015. Batch normalization: accelerating deep network training by reducing internal covariate shift. In: Proceedings of the 32nd International Conference on Machine Learning.





Lotte, F., Guan, C., Keng, A.K., 2009. Comparison of designs towards a subject-independent brain-computer interface based on motor imagery. In: 31st Annual International Conference of the IEEE Engineering in Medicine and Biology Society. EMBC '09).





Mahamune, R., Laskar, S.H., 2022. An automatic channel selection method based on the standard deviation of wavelet coefficients for motor imagery based brain-computer interfacing. Imag. Syst. Technol. 33 (2), 714-728.





Mao, W., et al., 2020. EEG dataset classification using CNN method. In: Journal of Physics: Conference Series. IOP Publishing.





Musallam, Y.K., et al., 2021. Electroencephalography-based motor imagery classification using temporal convolutional network fusion. Biomed. Signal Process Control 69, 102826.





Mzurikwao, D., et al., 2019. A channel selection approach based on convolutional neural network for multi-channel channel EEG motor imagery decoding. In: 2019 IEEE Second International Conference on Artificial Intelligence and Knowledge Engineering (AIKE).





Novi, Q., et al., 2007. Sub-band common spatial pattern (SBCSP) for brain-computer interface. Int. IEEE/EMBS Conf. Neural Eng. https://doi.org/10.1109/CNE.2007.369647.





Pawan, Dhiman, R., 2023. Electroencephalography channel selection based on pearson correlation coefficient for motor imagery-brain-computer interface. Measurement: Sensors 25.





Pfurtscheller, G., Silva, F.H.L.d., 1999. Event-related EEG/MEG synchronization and desynchronization: basic principles. Clin. Neurophysiol. 110 (11), 1842-1857.





Prechelt, L., 2002. Early stopping - but when? Neural Netw.: Tricks Trade 1524, 55-69. Pudil, P., Novovicova, J., Kittler, J., 1994. Floating search methods in feature selection. Pattern Recogn. Lett. 15 (11), 1119-1125.





Reunanen, J., Guyon, I., Elisseeff, A., 2003. Overfitting in making comparisons between variable selection methods. J. Mach. Learn. Res. 3, 1371-1382.





Rocha, A.d., Rocha, F., Arruda, L., 2013. A Neuromarketing Study of Consumer Satisfaction. Available at: SSRN 2321787.





Sadiq, M.T., et al., 2022. Exploiting pretrained CNN models for the development of an EEG-based robust BCI framework. Comput. Biol. Med. 143, 105242.





Schirrmeister, R.T., et al., 2017. Deep learning with convolutional neural networks for EEG decoding and visualization. Hum. Brain Mapp. 38 (11), 5391-5420.





Siskind, J.M., Pearlmutter, B.A., 2018. Divide-and-Conquer checkpointing for arbitrary programs with No user annotation. Optim. Methods Softw. 33 (4-6), 1288-1330. https://doi.org/10.1080/10556788.2018.1459621.





Sreeja, S.R., et al., 2017. Motor imagery EEG signal processing and classification using machine learning approach. In: 2017 International Conference in New Trends in Computing Sciences (ICTCS).





Srivastava, N., et al., 2014. Dropout: a simple way to prevent neural networks from overfitting. J. Mach. Learn. Res. 15 (1), 1929-1958.





Szegedy, C., et al., 2014. Going Deeper with Convolutions, vol. 1. Computer Vision and Pattern Recognition.





Thodoroff, P., Pineau, J., Lim, A., 2016. Learning robust features using deep learning for automatic seizure detection. In: 1st Machine Learning for Healthcare Conference.





Tibrewal, N., Leeuwis, N., Alimardani, M., 2022. Classification of motor imagery EEG using deep learning increases performance in inefficient BCI users. PLoS One 17 (7), e0268880.





Vijayendra, A., et al., 2018. A performance study of 14-channel and 5-channel EEG systems for real-time control of unmanned aerial vehicles (UAVs). 2018 Second IEEE International Conference on Robotic Computing (IRC).





Wang, Z.-M., Hu, S.-Y., Song, H., 2019. Channel selection method for EEG emotion recognition using normalized mutual information. IEEE Access 7.





Wolpaw, J.R., Wolpaw, e.W., 2012. Brain-computer interfaces: something new under the sun. Brain Comput. Interfac.: Princ. Pract. 3-12. https://doi.org/10.1093/acprof: oso/9780195388855.003.0001.





Xia, Y., et al., 2023. An adaptive channel selection and graph ResNet based algorithm for motor imagery classification. (IJACSA). Int. J. Adv. Comput. Sci. Appl. 14 (5).





Yuan, Y., et al., 2018. A novel channel-aware attention framework for multi-channel EEG seizure detection via multi-view deep learning. In: 2018 IEEE EMBS International Conference on Biomedical & Health Informatics (BHI).





Zeeshan Baig, M., Aslam, N., Shum, H.P.H., 2020. Filtering techniques for channel selection in motor imagery EEG applications: a survey. Artif. Intell. Rev. 53, 1207-1232.





Zhang, N., et al., 2023.  $\mathrm{CADC}++$ : advanced consensus-aware dynamic convolution for Co-salient object detection. IEEE Trans. Pattern Anal. Mach. Intell. 46 (5), 2741-2757.

