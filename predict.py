import os, time, json
# Set log level
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
import numpy as np

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

num_gpus = 1
batch_size = 16

def get_label():
    '''read label file'''
    label_map_path = "./data_train/labels.txt"
    label_map_file = open(label_map_path)
    label_map = {}
    for line_number, label in enumerate(label_map_file.readlines()):
        label_map[line_number] = label[:-1]
        line_number += 1
    label_map_file.close()

    return label_map

def get_dataset(filename):
    '''read tfrecord files'''
    def parse_exmp(serial_exmp):
        feats = tf.parse_single_example(serial_exmp, features={
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.string),
            'image/filepath': tf.FixedLenFeature([], tf.string),
            })
        image_data = tf.cast(feats['image/encoded'], tf.string)
        image_data = tf.image.decode_jpeg(image_data)
        image_data = tf.image.resize_images(image_data, [331, 331])
        image_data = tf.divide(image_data, 255)
        label = tf.cast(feats['image/class/label'], tf.string)
        filepath = tf.cast(feats['image/filepath'], tf.string)
        return image_data, label, filepath

    # extract data
    dataset = tf.data.TFRecordDataset(filename, num_parallel_reads=8)
    dataset = dataset.map(parse_exmp)
    dataset = dataset.batch(batch_size)
    return dataset

def predict(filename):
    # get label
    label_map = get_label()
    # read tfrecord file by tf.data
    dataset = get_dataset(filename)
    # dataset.apply(tf.contrib.data.prefetch_to_device("/gpu:0"))
    # load data
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()

    # session
    with tf.Session() as sess:
        tf.global_variables_initializer()
        t1 = time.time()
        res = []
        content = {}
        count = 0
        try:
            while True:
                [_image, _label, _filepath] = sess.run(fetches=features)
                _image = np.asarray([_image])
                _image = _image.reshape(-1, 331, 331, 3)

                for d in get_available_gpus():
                    print("Using device: {}".format(d))
                    with tf.device(d):
                        predictions = inference_session.run(output_layer, feed_dict={input_layer: _image})
                    predictions = np.squeeze(predictions)

                for i, pred in enumerate(predictions):
                    count += 1
                    overall_result = np.argmax(pred)
                    predict_result = label_map[overall_result].split(":")[-1]

                    if predict_result == 'unknown': continue

                    content['prob'] = str(np.max(pred))
                    content['label'] = predict_result
                    content['filepath'] = str(_filepath[i], encoding='utf-8')
                    res.append(content)

        except tf.errors.OutOfRangeError:
            t2 = time.time()
            print("{} images processed, average time: {}s".format(count, (t2-t1)/count))
    return res


if __name__ == '__main__':
    # print(get_available_gpus())
    # read model and load model graph
    model_dir = "./model"
    model = "nasnet_large_v1.pb"
    model_path = os.path.join(model_dir, model)
    model_graph = tf.Graph()
    with model_graph.as_default():
        with tf.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
            input_layer = model_graph.get_tensor_by_name("input:0")
            output_layer = model_graph.get_tensor_by_name('final_layer/predictions:0')

    # Session Config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    inference_session = tf.Session(graph = model_graph, config=config)

    # Initialize session
    initializer = np.zeros([1, 331, 331, 3])
    for d in get_available_gpus():
        print("Using device: {}".format(d))
        with tf.device(d):
            inference_session.run(output_layer, feed_dict={input_layer: initializer})

    # Prediction
    file_list = []
    processed_files = []

    for path, dir, files in os.walk('./model_output/processed_files'):
        for file in files:
            processed_files.append(file.split('_')[0]+'.tfrecord')

    print("Processed files: ")
    for f in processed_files:
        print('\t', f)

    while True:
        for path, dir, files in os.walk("./data_input"):
            for file in files:
                if file == '.DS_Store': continue
                if file in processed_files: continue
                print("Reading file {}".format(file))
                file_path = os.path.join('./data_input', file)
                file_list.append(file_path)
                res = predict(file_path)
                processed_files.append(file)

                with open('./model_output/processed_files/{}_{}_processed_files.json'.format(file.split('.')[0], model.split('.')[0]), 'w') as f:
                    f.write(json.dumps(processed_files))

                with open('./model_output/classify_result/{}_{}_classify_result.json'.format(file.split('.')[0], model.split('.')[0]), 'w') as f:
                    f.write(json.dumps(res, indent=4, separators=(',',':')))

        time.sleep(1)
