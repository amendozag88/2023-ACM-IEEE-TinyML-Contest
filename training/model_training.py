import os
import argparse
import numpy as np
import random
import tensorflow as tf
import datetime
from tensorflow import keras
from keras.optimizers import Adam
from helper_functions import stats_report, FB, loadCSV, txt_to_numpy
from scipy.signal import resample
from sklearn.metrics import confusion_matrix
from swa.tfkeras import SWA

class DataGenerator(tf.keras.utils.Sequence):
  def __init__(self, root_dir, indice_dir, mode, size, subject_id=None):
        self.root_dir = root_dir
        self.indice_dir = indice_dir
        self.size = size
        self.names_list = []

        #csvdata_all = loadCSV(os.path.join(self.indice_dir, mode + '_indice.csv'))
        csvdata_all = loadCSV(self.indice_dir + "/" + mode + '_indice.csv')

        for i, (k, v) in enumerate(csvdata_all.items()):
            # Check if the subject ID matches
            if subject_id is not None and k.startswith(subject_id):
                self.names_list.extend([f"{k} {filename}" for filename in v])
            elif subject_id is None:
                self.names_list.append(str(k) + ' ' + str(v[0]))

  def __len__(self):
    return len(self.names_list)

  def __getitem__(self, idx):
    text_path = self.root_dir + self.names_list[idx].split(' ')[0]

    if not os.path.isfile(text_path):
      print(text_path + 'does not exist')
      return None

    IEGM_seg = txt_to_numpy(text_path, self.size).reshape(1, self.size, 1)
    label = int(self.names_list[idx].split(' ')[1])
    sample = np.append(IEGM_seg, label)  #Giving the sample in this format instead of a dict as in IEGM_DataSET
    # sample = {'IEGM_seg': IEGM_seg, 'label': label}
    return sample

def save_model_to_tflite(path_and_filename, model_to_save):
    converter = tf.lite.TFLiteConverter.from_keras_model(model_to_save)
    tflite_model = converter.convert()
    with open(path_and_filename, 'wb') as f:
        f.write(tflite_model)
    print(f"Model saved as {path_and_filename}")

def save_quantized_model_to_tflite(path_and_filename, model_to_save):
    converter = tf.lite.TFLiteConverter.from_keras_model(model_to_save)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    with open(path_and_filename, 'wb') as f:
        f.write(tflite_quant_model)
    print(f"Quantized Model saved as {path_and_filename}")

def get_model():
  model = keras.Sequential([
      keras.layers.Input(shape=(625, 1)), 
      
      keras.layers.Conv1D(filters=2, kernel_size=5, strides=2, padding='valid', activation=None, use_bias=False),
      keras.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5, center=True, scale=True, 
        trainable=True, fused=False),
      keras.layers.ReLU(),

      keras.layers.Conv1D(filters=4, kernel_size=5, strides=2, padding='valid', activation=None, use_bias=False),
      keras.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5, center=True, scale=True, 
        trainable=True, fused=False),
      keras.layers.ReLU(),

      keras.layers.Conv1D(filters=6, kernel_size=4, strides=2, padding='valid', activation=None, use_bias=False),
      keras.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5, center=True, scale=True, 
        trainable=True, fused=False),
      keras.layers.ReLU(),

      keras.layers.Flatten(),
      keras.layers.Dropout(0.5),  #0.1
      keras.layers.Dense(8), # 5
      keras.layers.ReLU(),
      keras.layers.Dense(2),
  ])
  return model

def get_timestamp_for_filename():
    current_datetime = datetime.datetime.now()
    day, month = current_datetime.day, current_datetime.month
    hour, minute = current_datetime.hour, current_datetime.minute

    # Create a formatted string
    formatted_datetime = f"{month:02d}-{day:02d}_{hour:02d}-{minute:02d}"
    return formatted_datetime

def stretch(x, print_debug=False):
    x_stretched = np.zeros(x.shape)
    stretch = (random.random()/4)+0.05  #Always positive and from 0.05 to 0.3
    if print_debug:
      print("strech: ", str(stretch))
    l = int(625 * (1 + stretch))
    resampled_x = resample(x[:], l)
    if (l < 625):
        resampled_x_ = np.zeros(shape=(625, ))
        resampled_x_[:l] = resampled_x
    else:
        resampled_x_ = resampled_x[:625]
    x_stretched = resampled_x_
    return x_stretched

def scale(x, print_debug=False):
    x_scaled = np.zeros(x.shape)
    alpha = (random.random()-0.5)/4 
    if print_debug:
        print("Scale: ", (1+alpha))
    x_scaled = x*(1+alpha)
    return x_scaled

def add_noise(x, print_debug=False):
    scale  = x.max() * random.random() * 0.05
    noise = np.random.normal(0, scale, (len(x), 1))
    x_with_added_noise = x + noise
    return x_with_added_noise


def get_score_report_from_confusion_matrix (C):
    if C.shape == (2, 2):
        acc = (C[0][0] + C[1][1]) / (C[0][0] + C[0][1] + C[1][0] + C[1][1])

    if (C[1][1] + C[0][1]) != 0:
        precision = C[1][1] / (C[1][1] + C[0][1])
    else:
        precision = 0.0

    if (C[1][1] + C[1][0]) != 0:
        sensitivity = C[1][1] / (C[1][1] + C[1][0])
    else:
        sensitivity = 1.0

    FP_rate = C[0][1] / (C[0][1] + C[0][0])

    if (C[1][1] + C[1][0]) != 0:
        PPV = C[1][1] / (C[1][1] + C[1][0])
    else:
        PPV = 0.0

    NPV = C[0][0] / (C[0][0] + C[0][1])

    if (precision + sensitivity) != 0:
        F1_score = (2 * precision * sensitivity) / (precision + sensitivity)
    else:
        F1_score = 0.0

    if ((2 ** 2) * precision + sensitivity) != 0:
        F_beta_score = (1 + 2 ** 2) * (precision * sensitivity) / ((2 ** 2) * precision + sensitivity)
    else:
        F_beta_score = 0.0
    return acc, precision, sensitivity, FP_rate, PPV, NPV, F1_score, F_beta_score

def get_evaluation_score (model_to_evaluate, path_for_data, path_for_indices, print_debug=False):
    subject_data = {}
    all_metrics = []
    subjects_above_threshold = 0
    with open(path_for_indices+'test_indice.csv', 'r') as indice_file:
        for line in indice_file:
            label, filename = line.strip().split(',')
            subject_id = filename.split('-')[0]
            if subject_id not in subject_data:
                subject_data[subject_id] = []
            subject_data[subject_id].append((filename, label))
    confusion_matrix_scores = np.zeros (((len(subject_data.items())-1),2,2))         
    #Getting subject names to evaluate per subject_id
    for subject_idx, (subject_id, file_info_list) in enumerate(subject_data.items(), start=1):
        if subject_id == 'Filename':
            continue
        y_true_subject = []
        if (print_debug): 
            print(f"Processing subject {subject_id}")
        testX_for_this_subject = np.zeros((len(file_info_list),1250))
        for i in range(len(file_info_list)):
            filename, true_label = file_info_list[i]
            y_true_subject.append(true_label)
            testX_for_this_subject[i,:] = txt_to_numpy(path_for_data + filename, 1250)
        testX_for_this_subject = testX_for_this_subject[...,::2]
        testX_for_this_subject =  np.expand_dims(testX_for_this_subject, axis=2)
        y_pred_subject = model_to_evaluate.predict(testX_for_this_subject).argmax(axis=1)  
        
        # Perform calculations for each participant
        confusion_matrix_for_this_subject = confusion_matrix( np.array(y_true_subject, dtype=int), y_pred_subject)
        confusion_matrix_scores[subject_idx-2] = confusion_matrix_for_this_subject
        if (print_debug):
            print( confusion_matrix_scores[subject_idx-2] )
                
        acc, precision, sensitivity, FP_rate, PPV, NPV, F1_score, F_beta_score = get_score_report_from_confusion_matrix(confusion_matrix_for_this_subject)
        all_metrics.append([acc, precision, sensitivity, FP_rate, PPV, NPV, F1_score, F_beta_score])
        if F_beta_score > 0.95:
            subjects_above_threshold += 1
    #Now get the average FBeta and the G score
    subject_metrics_array = np.array(all_metrics)
    average_metrics = np.mean(subject_metrics_array, axis=0)
    acc, precision, sensitivity, FP_rate, PPV, NPV, F1_score, F_beta_score = average_metrics
    
    print("Final F_beta_score:", F_beta_score)
    proportion_above_threshold = subjects_above_threshold / len(all_metrics)
    g_score = proportion_above_threshold
    print("G Score:", g_score)

    challenge_partial_score = 70*F_beta_score + 30*g_score
    return challenge_partial_score, F_beta_score, g_score, confusion_matrix_scores  

def run_a_single_iteration(current_iteration, path_data, path_indices, SIZE):
    # Data aug setting
    add_noise_or_not = True
    aug_debug = False

    train_generator = DataGenerator(root_dir=path_data, indice_dir=path_indices, mode='train', size=SIZE)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_generator)
    train_dataset = train_dataset.shuffle(10).batch(len(train_generator))
    train_dataset = train_dataset.repeat()
    train_iterator = iter(train_dataset)

    test_generator = DataGenerator(root_dir=path_data, indice_dir=path_indices, mode='test', size=SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_generator)
    test_dataset = test_dataset.shuffle(10).batch(len(test_generator))
    test_dataset = test_dataset.repeat()
    test_iterator = iter(test_dataset)

    train_samples = train_iterator.get_next()

    #Downsampling the input
    x, y = train_samples[...,:-1:2], train_samples[...,-1]
    x = np.expand_dims(x, axis=2)

    test_samples = test_iterator.get_next()

    #Downsampling the test input
    x_test, y_test = test_samples[...,:-1:2], test_samples[...,-1]
    x_test = np.expand_dims(x_test, axis=2)
    
    x_aug = np.copy(x)
    y_aug = np.copy(y)
    stretch_or_not = random.random()
    scale_or_not = random.random()
    print(f"Length of x is {len(x)}")
    for i in range(len(x)):
        if stretch_or_not > 0.5:
            x_aug[i] = stretch(x_aug[i], print_debug=aug_debug)
        if scale_or_not > 0.5:
            x_aug[i] = scale(x_aug[i], print_debug=aug_debug)
        if add_noise_or_not:
            x_aug[i] = add_noise (x_aug[i], print_debug=aug_debug)
    x = np.concatenate((x, x_aug))
    y = np.concatenate((y, y_aug))

    start_epoch = 10
    swa = SWA(start_epoch=start_epoch, lr_schedule='cyclic', 
          swa_lr=0.0001, swa_lr2=0.0005, swa_freq=5, batch_size=args.batchsz, verbose=1)

    my_model = get_model()
    save_name = 'random_' + str(current_iteration) 
    checkpoint_filepath = './checkpoints/' + get_timestamp_for_filename() + save_name + '/'

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss', #'val_accuracy',
        mode='min', #'max',
        save_best_only=True)

    my_model.compile(optimizer=Adam(learning_rate=args.lr),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy']
                )
    print(f"Fitting model with {x.shape[0]} observations")
    my_model.fit(
    x,
    y,
    epochs=args.epoch,
    batch_size=args.batchsz,
    shuffle=True,
    validation_data=(x_test, y_test),
    callbacks=[model_checkpoint_callback, swa]
    )

    my_model.load_weights(checkpoint_filepath)
    score = my_model.evaluate(x_test, y_test)
    print('x_test shape:', x_test.shape)
    print('Model: ', save_name)
    print('acc', score[1])
    save_model_to_tflite('./checkpoint_models/' + get_timestamp_for_filename() + save_name + '.tflite', my_model)
    save_quantized_model_to_tflite('./checkpoint_models/' + get_timestamp_for_filename() + save_name + 'quant.tflite', my_model)

    challenge_partial_score, F_beta_score, g_score, confusion_matrix_scores = get_evaluation_score (my_model, path_data, path_indices+'/')
    
    return challenge_partial_score, F_beta_score, g_score, my_model
   
          
def main():
    # Hyperparameters
    BATCH_SIZE = args.batchsz
    BATCH_SIZE_TEST = args.batchsz
    LR = args.lr
    EPOCH = args.epoch
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices
    
    best_partial_score = 0.0
    number_of_iterations = 10

    for iteration in range(number_of_iterations):
        iteration_partial_score, iteration_FBeta, iteration_G, trained_model = run_a_single_iteration(iteration, path_data, path_indices, SIZE)
        if iteration_partial_score > best_partial_score:
            best_partial_score = iteration_partial_score
            save_model_to_tflite('./checkpoint_models/best_' + get_timestamp_for_filename() + str(iteration) + '.tflite', trained_model)
            save_quantized_model_to_tflite('./checkpoint_models/best_' + get_timestamp_for_filename() + str(iteration) + 'quant.tflite', trained_model)
            print(f"New partial score with FBeta {iteration_FBeta} and G {iteration_G} found! In iteration {iteration} with value {best_partial_score}")
    print(f"After running {number_of_iterations} times, best partial score is: {best_partial_score}")

     
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=100)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.0002)
    argparser.add_argument('--batchsz', type=int, help='total batchsz for traindb', default=32)
    argparser.add_argument('--cuda', type=int, default=0)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='./tinyml_contest_data_training/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices')

    args = argparser.parse_args()

    main()