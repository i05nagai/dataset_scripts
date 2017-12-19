

def set_tensorflow_config(
        allow_growth=True,
        num_gpu=1,
        per_process_gpu_memory_fraction=1.0):
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session

        gpu_options = tf.GPUOptions(allow_growth=allow_growth)
        gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
        visible_device_list = ",".join(str(i) for i in range(num_gpu))
        gpu_options.visible_device_list = visible_device_list

        config = tf.ConfigProto(gpu_options=gpu_options)
        session = tf.Session(config=config)
        set_session(session)
