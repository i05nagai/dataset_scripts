

def set_tensorflow_config(
        num_gpu=1,
        per_process_gpu_memory_fraction=1.0):
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session

        config = tf.ConfigProto()
        (config
         .gpu_options
         .per_process_gpu_memory_fraction) = per_process_gpu_memory_fraction
        visible_device_list = ",".join(str(i) for i in range(num_gpu))
        config.gpu_options.visible_device_list = visible_device_list
        set_session(tf.Session(config=config))
