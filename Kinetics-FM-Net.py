
def Kinetics_FM_Net(inputs_1D_N,inputs_2D_N):

  model_1=GRU(512,return_sequences=True)(inputs_1D_N)
  model_1=Dropout(0.3)(model_1)
  model_1=GRU(256,return_sequences=True)(model_1)
  model_1=Dropout(0.3)(model_1)
  model_1=Flatten()(model_1)
  
  model_2=GRU(512,return_sequences=True)(inputs_1D_N)
  model_2=Dropout(0.3)(model_2)
  model_2=GRU(256,return_sequences=True)(model_2)
  model_2=Dropout(0.3)(model_2)
  model_2=Flatten()(model_2)

  X=Conv2D(256, (5, 3), activation='relu',padding='same')(inputs_2D_N)
  X=BatchNormalization()(X)
  X=MaxPooling2D((2, 2))(X)
  X=Conv2D(256, (5, 3), activation='relu',padding='same')(X)
  X=BatchNormalization()(X)
  X=MaxPooling2D((2, 2))(X)
  X=Conv2D(512, (5, 3), activation='relu', padding='same')(X)
  X=BatchNormalization()(X)
  X=MaxPooling2D((2, 1))(X)
  X=Conv2D(512, (5, 3), activation='relu', padding='same')(X)
  X=BatchNormalization()(X)
  X=MaxPooling2D((2, 1))(X)

  X=Dense(64, activation='relu')(X)
  X=Dropout(0.05)(X)
  X=Dense(32,activation='relu')(X)
  X=Dropout(0.05)(X)

  X=Flatten()(X)

  X=concatenate([X,model_2])
  
  

  model_3=GRU(512,return_sequences=True)(inputs_1D_N)
  model_3=Dropout(0.3)(model_3)
  model_3=GRU(256,return_sequences=True)(model_3)
  model_3=Dropout(0.3)(model_3)
  model_3=Flatten()(model_3)

  CNN=Conv1D(filters=256, kernel_size=3, activation='relu',padding='same')(inputs_1D_N)
  CNN=BatchNormalization()(CNN)
  CNN=MaxPooling1D(pool_size=2)(CNN)
  CNN=Conv1D(filters=256, kernel_size=3, activation='relu',padding='same')(CNN)
  CNN=BatchNormalization()(CNN)
  CNN=MaxPooling1D(pool_size=2)(CNN)
  CNN=Conv1D(filters=512, kernel_size=3, activation='relu',padding='same')(CNN)
  CNN=BatchNormalization()(CNN)
  CNN=MaxPooling1D(pool_size=2)(CNN)
  CNN=Conv1D(filters=512, kernel_size=3, activation='relu',padding='same')(CNN)
  CNN=BatchNormalization()(CNN)
  CNN=MaxPooling1D(pool_size=2)(CNN)

  CNN=Dense(64, activation='relu')(CNN)
  # CNN=Dropout(0.05)(CNN)
  CNN=Dense(32, activation='relu')(CNN)
  # CNN=Dropout(0.05)(CNN)
  CNN=Flatten()(CNN)

  CNN=concatenate([CNN,model_3])

  output_GRU=Dense(num_pred*w,activation='linear')(model_1)
  output_GRU=Reshape(target_shape=(w,num_pred))(output_GRU)
  output_C2=Dense(num_pred*w,activation='linear')(X)
  output_C2=Reshape(target_shape=(w,num_pred))(output_C2)
  output_C1=Dense(num_pred*w,activation='linear')(CNN)
  output_C1=Reshape(target_shape=(w,num_pred))(output_C1)

  output_GRU_1=Dense(128,activation='relu')(output_GRU)
  output_GRU_1=Dense(num_pred,activation='sigmoid')(output_GRU_1)
  output_GRU_2 = tf.keras.layers.Multiply()([output_GRU, output_GRU_1])
  
  output_C2_1=Dense(128,activation='relu')(output_C2)
  output_C2_1=Dense(num_pred,activation='sigmoid')(output_C2_1)
  output_C2_2 = tf.keras.layers.Multiply()([output_C2, output_C2_1])

  output_C1_1=Dense(128,activation='relu')(output_C1)
  output_C1_1=Dense(num_pred,activation='sigmoid')(output_C1_1)
  output_C1_2 = tf.keras.layers.Multiply()([output_C1, output_C1_1])
  

  weight=output_GRU_1+output_C2_1+output_C1_1
  
  output_GRU_2 = tf.keras.layers.Multiply()([output_GRU, output_GRU_1])
  output_C2_2 = tf.keras.layers.Multiply()([output_C2, output_C2_1])
  output_C1_2 = tf.keras.layers.Multiply()([output_C1, output_C1_1])
  

  output = output_GRU_2+output_C1_2+output_C2_2
  

  return (output_C1,output_C2,output_GRU,output)

  
def Kinetics_FM_Net_pcc(inputs_1D_N,inputs_2D_N):

  model_1=GRU(512,return_sequences=True)(inputs_1D_N)
  model_1=Dropout(0.25)(model_1)
  model_1=GRU(256,return_sequences=True)(model_1)
  model_1=Dropout(0.25)(model_1)
  model_1=Flatten()(model_1)
  
  model_2=GRU(512,return_sequences=True)(inputs_1D_N)
  model_2=Dropout(0.3)(model_2)
  model_2=GRU(256,return_sequences=True)(model_2)
  model_2=Dropout(0.3)(model_2)
  model_2=Flatten()(model_2)

  X=Conv2D(256, (5, 3), activation='relu',padding='same')(inputs_2D_N)
  X=BatchNormalization()(X)
  X=MaxPooling2D((2, 2))(X)
  X=Conv2D(256, (5, 3), activation='relu',padding='same')(X)
  X=BatchNormalization()(X)
  X=MaxPooling2D((2, 2))(X)
  X=Conv2D(512, (5, 3), activation='relu', padding='same')(X)
  X=BatchNormalization()(X)
  X=MaxPooling2D((2, 1))(X)
  X=Conv2D(512, (5, 3), activation='relu', padding='same')(X)
  X=BatchNormalization()(X)
  X=MaxPooling2D((2, 1))(X)

  X=Dense(64, activation='relu')(X)
  # X=Dropout(0.05)(X)
  X=Dense(32,activation='relu')(X)
  # X=Dropout(0.05)(X)

  X=Flatten()(X)
  X=concatenate([X,model_2])
  
  
  model_3=GRU(512,return_sequences=True)(inputs_1D_N)
  model_3=Dropout(0.3)(model_3)
  model_3=GRU(256,return_sequences=True)(model_3)
  model_3=Dropout(0.3)(model_3)
  model_3=Flatten()(model_3)

  CNN=Conv1D(filters=256, kernel_size=3, activation='relu',padding='same')(inputs_1D_N)
  CNN=BatchNormalization()(CNN)
  CNN=MaxPooling1D(pool_size=2)(CNN)
  CNN=Conv1D(filters=256, kernel_size=3, activation='relu',padding='same')(CNN)
  CNN=BatchNormalization()(CNN)
  CNN=MaxPooling1D(pool_size=2)(CNN)
  CNN=Conv1D(filters=512, kernel_size=3, activation='relu',padding='same')(CNN)
  CNN=BatchNormalization()(CNN)
  CNN=MaxPooling1D(pool_size=2)(CNN)
  CNN=Conv1D(filters=512, kernel_size=3, activation='relu',padding='same')(CNN)
  CNN=BatchNormalization()(CNN)
  CNN=MaxPooling1D(pool_size=2)(CNN)

  CNN=Dense(64, activation='relu')(CNN)
  # CNN=Dropout(0.05)(CNN)
  CNN=Dense(32, activation='relu')(CNN)
  # CNN=Dropout(0.05)(CNN)
  CNN=Flatten()(CNN)

  CNN=concatenate([CNN,model_3])

  output_GRU=Dense(num_pred*w, activation='linear')(model_1)
  output_GRU=Reshape(target_shape=(w,num_pred))(output_GRU)
  output_C2=Dense(num_pred*w, activation='linear')(X)
  output_C2=Reshape(target_shape=(w,num_pred))(output_C2)
  output_C1=Dense(num_pred*w, activation='linear')(CNN)
  output_C1=Reshape(target_shape=(w,num_pred))(output_C1)

  
  output_GRU_1=Dense(128,activation='relu')(output_GRU)
  output_GRU_1=Dropout(0.4)(output_GRU_1)
  output_GRU_1=Dense(num_pred,activation='sigmoid')(output_GRU_1)
  output_GRU_2 = tf.keras.layers.Multiply()([output_GRU, output_GRU_1])
  
  output_C2_1=Dense(128,activation='relu')(output_C2)
  output_C2_1=Dropout(0.4)(output_C2_1)
  output_C2_1=Dense(num_pred,activation='sigmoid')(output_C2_1)
  output_C2_2 = tf.keras.layers.Multiply()([output_C2, output_C2_1])

  output_C1_1=Dense(128,activation='relu')(output_C1)
  output_C1_1=Dropout(0.4)(output_C1_1)
  output_C1_1=Dense(num_pred,activation='sigmoid')(output_C1_1)
  output_C1_2 = tf.keras.layers.Multiply()([output_C1, output_C1_1])
  

  weight=output_GRU_1+output_C2_1+output_C1_1
  
  output_GRU_2 = tf.keras.layers.Multiply()([output_GRU, output_GRU_1])
  output_C2_2 = tf.keras.layers.Multiply()([output_C2, output_C2_1])
  output_C1_2 = tf.keras.layers.Multiply()([output_C1, output_C1_1])
  
  output = output_GRU_2+output_C1_2+output_C2_2
    
  return (output_C1,output_C2,output_GRU,output)
