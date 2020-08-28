####packages
import numpy as np
import pandas as pd
import os
import re
import scipy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors
import PIL
from matplotlib import colors
from IPython.display import IFrame
from datetime import datetime
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Reshape
from keras.utils.np_utils import to_categorical
from keras.callbacks import History 
from keras.models import Model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

def read_sequence_file(fpath):
    d = {}
    lines = []
    with open(fpath, 'rt') as f:
        lines = f.read().splitlines()  # to exclude newline symbols
    seq_dict ={key: value 
               for key, value in zip(lines[::2],  # <- all odd lines
                                                lines[1::2]  # <- all even lines
                                               )
              }
    return seq_dict

def make_iteration_first_level_key(key):
    """
    returns 2 level representaion of key
    here we can define to have 'it' as top level and all other stuff as 2nd level
    '>it0pop3ind53locus0' -> ('it0000', 'pop0003ind0053locus0000')
    """
    key_dict = split_key(key)
    level1 = 'it%04d' % key_dict['it']
    level2 = 'pop%04dind%04dlocus%04d' % (key_dict['pop'], key_dict['ind'], key_dict['locus'])
    return (level1, level2)

def split_key(key):
    """
    returns dict representation of key
    '>it0pop3ind53locus0' -> {'ind': 53, 'it': 0, 'pop': 3, 'locus': 0}
    """
    regex = re.compile(r"([^\W\d_]+)(\d+)")
    key_dict = {part: int(number) for part, number in regex.findall(key)}
    assert sorted(list(key_dict.keys())) == ['ind', 'it', 'locus', 'pop']
    return key_dict
      
def resize_array(input_array,height,width):
    new_proportions=(height,width)
    input_array=rescore_from_array_to_image(input_array)
    img = PIL.Image.fromarray(input_array)
    resized_img = img.resize(new_proportions, PIL.Image.BICUBIC)
    resized_array=rescore_from_image_to_array(resized_img)
    return(resized_array)

def reduce_3D_array(input_array_3d, height,width):
    array_out=np.zeros([input_array_3d.shape[0],width,height]) ####this is confusing - to be addressed
    for i in range(input_array_3d.shape[0]):
        array_out[i,:,:]=resize_array(input_array_3d[i,:,:],height,width)
    return array_out
    
def plot_cnn_loop_new(array,number,ax_base_shape,standard_colours,image_resize, shuffle=False):
    z=array[number,:,:]
    z=rescore_from_array_to_image(z)
    if shuffle:
        # shuffle tiles
        z = shuffle_tiles_of_array(z, (z.shape[0], z.shape[1] // 4), random_state=None)
    img = PIL.Image.fromarray(z)
    img2 = img.resize(image_resize, PIL.Image.BICUBIC)
    imgext = [0,ax_base_shape[1],0,ax_base_shape[0]]
    if standard_colours=="on":   
        cmap = colors.ListedColormap(["white", "lightblue", "blue","green"]); 
        bounds=[-1,57,121,184,255]; 
        norm = colors.BoundaryNorm(bounds, 4)    
    else:
        cmap = "Blues_r"
        norm=colors.NoNorm()
    plt.imshow(np.array(img2), cmap=cmap,interpolation='none',extent=imgext,norm=norm)
    
def rescore_from_array_to_image(input_array):
    input_array=np.true_divide(input_array,4) ###to be adjusted - but genomic array has max of 4
    input_array = (input_array * 255).astype(np.uint8)
    return(input_array)

def rescore_from_image_to_array(input_array):
    input_array = np.true_divide(input_array,255)
    input_array = np.round((input_array * 4))
    return(input_array)
            
def shuffle_tiles_of_array(array, tile_size_tuple, random_state=None):
    rand_state = np.random.RandomState(seed=random_state)
    i,j = array.shape
    k,l = tile_size_tuple
    esh = i//k,k,j//l,l
    bc = esh[::2]
    sh1,sh2 = np.unravel_index(rand_state.permutation(bc[0]*bc[1]),bc)
    ns1,ns2 = np.unravel_index(np.arange(bc[0]*bc[1]),bc)
    out = np.empty_like(array)
    out.reshape(esh)[ns1,:,ns2] = array.reshape(esh)[sh1,:,sh2]
    return out

def sort_haplo(data, sortindex, dropcolumns=False):
    # remove 2 first columns Index and Location
    if dropcolumns==True:
        data.drop(data.columns[[0,1]],axis=1,inplace=True)
    else: 
        pass
    # convert from pandas dataframe to numpy array
    data2 = np.array(data)

    ###sort 
    sortIds,bestLength = sort_by_common_substring(data2, sortindex)
    strLen = data2.shape[0] # length of "string"
    strNums = data2.shape[1] # number of "strings"
    n = strLen//2 #midpoint

    img = np.zeros((strLen, strNums), dtype=int)
    centrcharSorted = data2[n,sortIds]
    lenSorted = bestLength[sortIds]
    for i in range(strNums):
        startPos = (strLen-lenSorted[i])//2
        img[startPos:startPos+lenSorted[i],i] = [centrcharSorted[i]]*lenSorted[i]
    return(img)

def haplo_squeeze_and_sort(unsorted_array,sortindex):
    a=np.squeeze(unsorted_array) #[0:2,:,:]
    filter_array=np.zeros(a.shape)
    for i in range(a.shape[0]):
        #print(i,"iteration")        
        b=a[i,:,:]
        filter_array[i,:,:]=sort_haplo(b,sortindex) ##"two_pop" "country"
    return(filter_array)

def harmonize_images_to_01code(input_array_in):
    input_array=input_array_in.copy()
    input_array[input_array>1]=1
    return input_array

def performance_curve_accuracy(history,title,datestr):
    plotdict=history    
    plt.plot(plotdict['acc'],color="blue")
    plt.plot(plotdict['val_acc'],color="orange")
    plt.title('model accuracy'+"_"+title)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower left')
    plt.savefig('model loss_graph'+"_"+title+"_"+datestr)
    plt.ylim(0,1)
    plt.show()

def performance_curve_loss(history,title,datestr):
    plotdict=history    
    plt.plot(plotdict['loss'],color="blue")
    plt.plot(plotdict['val_loss'],color="orange")
    plt.title('model loss'+"_"+title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valiation'], loc='upper left')
    plt.savefig('model loss_graph'+"_"+title+"_"+datestr)
    plt.ylim(0,1)
    plt.show()

def sort_by_common_substring(data2, mainGroup):
    strLen = data2.shape[0] # length of "string"
    strNums = data2.shape[1] # number of "strings"
    similarity_matrix=np.zeros((strNums,strNums), dtype=int)
    match_position=np.zeros((strNums,), dtype=int)
    match_pos_matrix=np.zeros((strNums,strNums), dtype=int)
    totalMatch=np.zeros((strNums), dtype=int)

    n = strLen//2 #midpoint
    for i in range(strNums):
        #print(i)

        #b=np.array(data.ix[:,i] == data2.transpose()).astype(int).transpose()
        b = (data2[:,i].reshape((strLen,1)) == data2).astype(int)
        botPart=b[n:,:].argmin(0) # calculation botPart includes "central character"
        topPart=b[n-1::-1,:].argmin(0)
        # correcting data for substrings which contains all 1 (if b[?,i] contains all 1 argmin(0) returns 0 value,
        #     need to change to length of corresponing part(top/boottom)/substring)
        botPart[(botPart==0) & (b[n,:]==1)] = strLen-n
        topPart[(topPart==0) & (b[n-1,:]==1)] = n
        #match_position = n-topPart
        match_position = strLen-n-botPart
        similarity_vector=topPart+botPart
        similarity_vector[i] = 0
        totalMatch[i] = np.sum(similarity_vector[(data2[n,i]==data2[n,:]) & (mainGroup[i]==mainGroup)])
        #print(similarity_vector)
        #similarity_vector=b[n:,:].argmin(0)+b[n-1::-1,:].argmin(0)
        similarity_vector=similarity_vector[None,:] ##adjust vector
        similarity_matrix[i,:]=similarity_vector
        match_pos_matrix[i,:]=match_position[None,:]

    #DEBUG OUTPUT
    if strNums<=20:
        print("similarity matrix:")
        for i in range(strNums):
            print(' '.join([str(v) for v in similarity_matrix[i]]) + ' | '+str(totalMatch[i]))

    # b_extended = np.vstack([b, similarity_vector]) ###add as last row
    # c=b_extended[:, np.argsort( b_extended[-1] ) ] ### sort based on last vector #last row needs to be deleted still
    #numDig = math.ceil(math.log10(strLen))+1 # number of magnitude of string length +1 (number of digits to write string length)
    rowMeasure = np.empty((strNums,))
    bestLength = np.empty((strNums,), dtype=int)
    bestPos = np.empty((strNums,), dtype=int)

    # select "main" string (withing each "central character group")
    centrChars = np.unique(data2[n])
    mainGroupChars = np.unique(mainGroup)
    # mainStr contains index of "main" string for each "central character group"
    for i in range(len(centrChars)):
        cc = centrChars[i]
        for j in range(len(mainGroupChars)):
            groupIds = (data2[n]==cc) & (mainGroup==mainGroupChars[j])
            if np.any(groupIds):
                maxVal = np.max(totalMatch[groupIds])
                mainStr = np.where((data2[n]==cc) & (mainGroup==mainGroupChars[j]) & (totalMatch==maxVal))[0][0]
                bestLength[groupIds] = similarity_matrix[mainStr, groupIds]
                bestPos[groupIds] = match_pos_matrix[mainStr, groupIds]
                bestPos[mainStr] = 0
                bestLength[mainStr] = strLen
    """rowMeasure = data2[n] * (10.0**(2*numDig)) + bestLength*(10.0**numDig) + (strLen-bestPos)
    sortIds = rowMeasure.argsort()[::-1]
    print "rowMeasure",rowMeasure
    print "sortIds",sortIds"""

    sortIds2 = np.lexsort((strLen-bestPos, bestLength, data2[n], mainGroup))[::-1]
    #print "sortIds2",sortIds2

    #DEBUG OUTPUT
    if strNums<=20:
        print(data2[:,sortIds2])

    #DEBUG OUTPUT
    if strNums<=20:
        for i in range(strNums):
            i2 = sortIds2[i]
            s = (data2[:,i2])[::-1]
            print(''.join([str(v) for v in s])+' | '+str(bestPos[i2])+' '+str(bestLength[i2]))
            
    return (sortIds2, bestLength)

def plot_page_of_images_simulated_images_new(ax_base_shape,filename,array,
                                         source_simulations,divergence_midpoint,
                                         standard_colours,annotation=False,image_resize=1938,
                                         shuffle=False,PLT_RCPARAMS_DICT=0):
    total_images_toplot=array.shape[0]
    print(total_images_toplot)
    with PdfPages(filename) as pdf:
        with plt.rc_context(PLT_RCPARAMS_DICT):
            fig = plt.figure(figsize=(height_image,width_image),dpi=dpi_image) 
            for i in range(total_images_toplot):
                if i % numPlotsOnPage == 0:
                    if i!=0:
                        plt.tight_layout()
                        pdf.savefig()
                        print("hitcloeout")
                        plt.close()
                    fig = plt.figure(figsize=(height_image,width_image),dpi=15)
                    print("hit-create-new-page")
                a=fig.add_subplot(images_per_row,number_of_columns,i%numPlotsOnPage+1)
                a.set_xlim(0, ax_base_shape[1])
                a.set_ylim(0, ax_base_shape[0])

                imgplot = plot_cnn_loop_new(array,i,ax_base_shape,standard_colours,image_resize, shuffle=shuffle)
                #imgplot = plt.matshow(np.random.random((50,50)));plt.show()
                if annotation==True:
                    title="%d_%s_%.5f" % (i, source_simulations[i], divergence_midpoint[i])
                else:
                    title=str("no title")
                a.set_title(title)
            plt.tight_layout()
            pdf.savefig()
            plt.close() 

def create_crosstab(df,criterion):
    df = df[df['source']==criterion]
    print("accuracy is {}".format(1-np.true_divide(np.sum(df["acc"]!=0),df.shape[0])))
    confusion_matrix = pd.crosstab(df['label'], df['pred'], rownames=['Actual'], colnames=['Predicted'])
    print (confusion_matrix)

def create_partitions_for_prediction(df):
    interim_list=list(df.source.unique()) ###get all possible outcomes 
    interim_list.remove("sel_neutral") ###remove sel_neutral
    df["test_partitions"]=df.source
    df.loc[df.source=="sel_neutral","test_partitions"]=np.random.choice(interim_list, np.sum(df.source=="sel_neutral"))
    return(df)

def accuracy_metrics(cm):
    TP = cm[1][1]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[0][0]

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = np.true_divide(TP,(TP+FN))
    # Specificity or true negative rate
    TNR = np.true_divide(TN,(TN+FP))
    # Precision or positive predictive value
    PPV = np.true_divide(TP,(TP+FP))
    # Negative predictive value
    NPV = np.true_divide(TN,(TN+FN))
    # Fall out or false positive rate
    FPR = np.true_divide(FP,(FP+TN))
    # False negative rate
    FNR = np.true_divide(FN,(TP+FN))
    # False discovery rate
    FDR = np.true_divide(FP,(TP+FP))
    ACC = np.true_divide(TP+TN,TP+FP+FN+TN)
    return(TPR,TNR,PPV,NPV,FPR,FNR,FDR,ACC)

def accuracy_and_cm(test_values,predicted_values):
    cm = confusion_matrix(test_values,predicted_values)
    print()
    print(cm); print()
    TPR,TNR,PPV,NPV,FPR,FNR,FDR,ACC = accuracy_metrics(cm)
    print("TPR = %0.2f, TNR = %0.2f, ACC = %0.2f" % (TPR*100,TNR*100,ACC*100))
    return(ACC,TPR,TNR)

def update_dataframe_2(df,criterion):
    test_values,predicted_values=subset_and_get_test_and_predicted_values(df,criterion,use_probability_output=0)
    n_samples=len(test_values)
    ACC,TPR,TNR=accuracy_and_cm(test_values,predicted_values)
    print(n_samples,ACC,TPR,TNR,"final values")
    return n_samples,100*ACC,100*TPR,100*TNR

def obtain_roc_values(test_values,predicted_values):
    fpr, tpr, _ = roc_curve(test_values,predicted_values)
    roc_auc = auc(fpr, tpr)
    return(fpr, tpr,roc_auc)

def subset_and_get_test_and_predicted_values(dg,criterion,use_probability_output=0):
    df=dg.copy()
    if criterion=="all":
        pass
    else:
        df = df[df['test_partitions']==criterion]
    #predicted_values=[x[0] for x in df['pred_prob']]
    if use_probability_output==0:
        predicted_values=df['pred']
    else:
        predicted_values=df['pred_prob'].str.strip('[]').astype(float)
    #print(predicted_values)
    test_values=df.label
    #print(test_values)
    return test_values,predicted_values

def draw_roc_curve(df,criterion,title,color):
    test_values,predicted_values=subset_and_get_test_and_predicted_values(df,criterion,use_probability_output=1)
    fpr, tpr,roc_auc =obtain_roc_values(test_values,predicted_values)
    plt.plot(fpr, tpr, color=color,lw=2, label=title % roc_auc)

def simulation_performance_dataframe(df_predictions_sim,datestr,write_out):
    interim_list=list(df_predictions_sim.source.unique()) ###get all possible outcomes 
    interim_list.remove("sel_neutral");interim_list.append("all") ###remove sel_neutral, add "all"
    columns=["n_samples","Accuracy","TPR","TNR","AUC"]
    df_roc = pd.DataFrame(index=interim_list, columns=columns)
    df_roc = df_roc.fillna(0.0) # with 0s rather than NaNs
    for criterion in interim_list:     ###populate dataframe
        print(criterion)
        df_roc.loc[criterion]["n_samples"],df_roc.loc[criterion]["Accuracy"],df_roc.loc[criterion]["TPR"],df_roc.loc[criterion]["TNR"]=update_dataframe_2(df_predictions_sim,criterion)  
    df_roc.round(1) ###round numbers
    df_roc.reindex(['sel_strong', "sel_historic", 'sel_weak',"sel_partial","all"])
    title_graph="SFS_simulation_results_"+datestr+".csv"
    if write_out==1:
        df_roc.to_csv(title_graph)
    return(df_roc)

def plot_roc_curve(df_predictions_sim,datestr,write_out):
    plt.figure()
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    draw_roc_curve(df_predictions_sim,"all","'all (AUC = %0.3f)'","aqua")
    draw_roc_curve(df_predictions_sim,"sel_weak","'weaker (AUC = %0.3f)'","red")
    draw_roc_curve(df_predictions_sim,"sel_historic","'historic (AUC = %0.3f)'","red")
    draw_roc_curve(df_predictions_sim,"sel_partial","'partial (AUC = %0.3f)'","green")
    draw_roc_curve(df_predictions_sim,"sel_strong","'stronger (AUC = %0.3f)'","blue")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    title_graph="SFS_simulation_roc_curves_"+datestr+".png"
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')#; plt.title(title_graph)
    plt.legend(loc="lower right")
    if write_out==1:
        plt.savefig(title_graph)
    plt.show()

def create_predictions_dataframe(id_test,source_simulations_test,y_test,divergence_midpoint_test,x_test_input,x_test_classes_input,show_mispredictions,write_out):
    df_predictions_sim=pd.DataFrame(zip(id_test,source_simulations_test,y_test,divergence_midpoint_test,x_test_input,x_test_classes_input))
    df_predictions_sim.columns=["id","source","label","div_mid","pred_prob","pred"]
    df_predictions_sim["acc"]=df_predictions_sim["label"]-df_predictions_sim["pred"]
    print(1-np.true_divide(np.sum(df_predictions_sim["acc"]!=0),df_predictions_sim.shape[0]),"accuracy")
    if show_mispredictions==1: 
        df_predictions_sim[df_predictions_sim["acc"]!=0]
    df_predictions_sim['pred'] = df_predictions_sim['pred'].apply(lambda x: x[0])
    print(pd.crosstab(df_predictions_sim['label'], df_predictions_sim['pred'], rownames=['Actual'], colnames=['Predicted']))
    if write_out==1:
        df_predictions_sim.to_csv("df_predictions_sim.csv")
    print("The predictive accuracy of the model on the test sest is {} percent".format(100*np.true_divide(np.sum(df_predictions_sim["label"]==df_predictions_sim["pred"]),df_predictions_sim.shape[0])))
    return(df_predictions_sim)
           
def load_multi_ancestral_sequence(origin_fpath_list):
    return np.vstack([load_single_ancestral_sequence(fpath) for fpath in origin_fpath_list])
   
def calc_divergence_for_array(array_no_enc, multi_ancestral_sequence):
    divergence = [np.mean(array_slice.T != ancestral_sequence, axis=0)
                  for array_slice, ancestral_sequence in zip(array_no_enc, multi_ancestral_sequence)]
    return np.array(divergence)

def slice_divergence_at_point(array_divergence, point=500):
    return array_divergence[:, point]

def load_single_ancestral_sequence(origin_fpath):
    """
    returns all lines from origin_fpath next to line '>locus_0'
    """
    
    with open(origin_fpath, 'rt') as f:
        lines = f.read().splitlines()
    locus_line_nums = [i + 1 for i, line in enumerate(lines) 
                       if line == '>locus_0']
    return np.array([list(lines[line_num]) for line_num in locus_line_nums], 
                    ndmin=2)

def load_multi_ancestral_sequence(origin_fpath_list):
    return np.vstack([load_single_ancestral_sequence(fpath) for fpath in origin_fpath_list])

def make_iteration_first_level_dataframe(sequence_dict, encode=True):
    res = pd.DataFrame({make_iteration_first_level_key(key): list(value) 
                      for key, value in sequence_dict.items()})
    res.sort_index(axis=1,  # sort columns order
                   level=[0, 1], # sort both levels
                   ascending=[True, True], 
                   inplace=True)
    if encode:
        res.replace("A",1,inplace=True)
        res.replace("C",2,inplace=True)
        res.replace("G",3,inplace=True)
        res.replace("T",4,inplace=True)
    return res

def load_multi_iteration_sequence(list_of_fpath, encode=True):
    """
    returns 3D array of shape 
    (total iterations in all files in list_of_fpath, number of identifiers, number of positions)
    """
    arrays = []
    for fpath in list_of_fpath:
        sequence_dict = read_sequence_file(fpath)
        iteration_level_df = make_iteration_first_level_dataframe(sequence_dict, encode=encode)
        iteration_level_df_stacked = (iteration_level_df
                                      .stack(level=0)
                                      .swaplevel()
                                      .sort_index()
                                     )
        array3d = np.array(list(iteration_level_df_stacked
                                .groupby(level=0)
                                .apply(lambda _df: _df.values)))
        arrays.append(array3d)
    return  np.vstack(arrays)
