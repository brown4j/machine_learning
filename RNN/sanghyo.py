from optparse import Option
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Activation, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# 데이터 불러오기
inputs = pd.read_csv('./data/train_input.csv')
outputs = pd.read_csv('./data/train_output.csv')


# nan 제거
inputs['강우감지'] = inputs['강우감지'].fillna(value=0)
inputs['지습'] = inputs['지습'].fillna( inputs['지습'].mean() )

# inputs['외부온도'] = inputs['외부온도'].fillna(value=0)
# inputs['외부풍향'] = inputs['외부풍향'].fillna(value=0)
inputs['외부풍속'] = inputs['외부풍속'].fillna( inputs['외부풍속'].mean() )

inputs = inputs.dropna(axis=1)
# inputs.to_csv("./1.csv") 

# 주차 정보 수치 변환
inputs['주차'] = [int(i.replace('주차', "")) for i in inputs['주차']]


# scaler
input_scaler = MinMaxScaler()
output_scaler = MinMaxScaler()
dropped_list = []

# scaling
input_sc = input_scaler.fit_transform(inputs.iloc[:, 3:].to_numpy())
output_sc = output_scaler.fit_transform(outputs.iloc[:, 3:].to_numpy())
# outputs.iloc[:, 3:].to_csv("./2.csv") 


# 입력 시계열화
input_ts = []
for i in outputs['Sample_no']:
    sample = input_sc[inputs['Sample_no'] == i]
    if len(sample) < 1:
        continue
    elif len(sample) < 6:
        t0 = sample[0]
        t1 = sample[-1]
        sample = np.append(np.zeros((1, sample.shape[-1])) + t0, sample, axis=0)
        sample = np.append(np.zeros((6-len(sample), sample.shape[-1])) + (t0+t1)/2, sample, axis=0)
        sample = np.append(np.zeros((1, sample.shape[-1])) + t1, sample, axis=0)
        print(i, len(sample))
    elif len(sample) < 7:
        sample = np.append(np.zeros((7-len(sample), sample.shape[-1])) + sample[-1], sample, axis=0)

    sample = np.expand_dims(sample, axis=0)
    input_ts.append(sample)
input_ts = np.concatenate(input_ts, axis=0)

# print(dropped_list)
# for i in dropped_list :
    # output_sc.delete(output_sc[output_sc["Sample_no"] == i].index)

# 셋 분리
train_x, val_x, train_y, val_y = train_test_split(input_ts, output_sc, test_size=0.15, shuffle=True, random_state=0)
input_shape = (train_x.shape[1], train_x.shape[2])

# 모델 정의
def deep_lstm():
    # https://buomsoo-kim.github.io/keras/2019/07/29/Easy-deep-learning-with-Keras-20.md/
    model = Sequential()
    model.add(Bidirectional(LSTM(256, return_sequences = True), input_shape=input_shape))
    model.add(Bidirectional(LSTM(256, return_sequences = False)))
    # model.add(LSTM(512, input_shape = (train_x.shape[1], train_x.shape[2]), return_sequences = True))
    # model.add(LSTM(20, return_sequences = True))
    # model.add(LSTM(128, return_sequences = True))
    # model.add(LSTM(512, return_sequences = False))
    model.add(Dense(3))
    model.add(Activation('tanh'))    
    return model

def create_model():
    x = Input(shape=input_shape)
    # LSTM의 입력인 x는 기본적으로 3차원 구조를 갖는다. 첫 번째 차원은 데이터 개수다 (sample 개수, batch 개수라고도 한다). 
    # 이 경우는 1이다. 두 번째 차원은 시간축의 차원이다 (time step size라고도 한다). 이 경우는 x가 1,2,3,4,5로 5개가 연속되므로 5이다. 
    # LSTM이 5번 반복 (recurrent)되는 것이다. 마지막 차원은 LSTM 입력층에 입력되는 데이터의 개수다 (feature 개수라고도 한다). 
    # 이 경우는 LSTM의 각 스텝에 데이터가 한 개씩 들어가므로 1이다. 따라서 입력 데이터의 차원 (input_shape, batch_shape라 한다)은 (1, 5, 1)이다.

    l1 = LSTM(256)(x)
    out = (Dense(3, activation='tanh'))(l1)
    return Model(inputs=x, outputs=out)

# model = deep_lstm()
model = create_model()

if True :
    model.summary()
    checkpointer = ModelCheckpoint(monitor='val_loss', filepath='baseline.h5',
                                verbose=1, save_best_only=True, save_weights_only=True)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mse'])

    # 학습
    hist = model.fit(train_x, train_y, batch_size=50, epochs=300, validation_data=(val_x, val_y), callbacks=[checkpointer])
    #default batchsize = 32?

    # loss 히스토리 확인
    fig, loss_ax = plt.subplots()
    loss_ax.plot(hist.history['loss'], 'r', label='loss')
    loss_ax.plot(hist.history['val_loss'], 'g', label='val_loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend()
    plt.ylim([0.0, 0.002])
    # plt.axhline()
    plt.grid(True)
    plt.title('Training loss - Validation loss plot')
    plt.show()

# 저장된 가중치 불러오기
model.load_weights('baseline.h5')

# 테스트셋 전처리 및 추론
test_inputs = pd.read_csv('./data/test_input.csv')
output_sample = pd.read_csv('./data/answer_sample.csv')

test_inputs = test_inputs[inputs.columns]
test_inputs['주차'] = [int(i.replace('주차', "")) for i in test_inputs['주차']]

# 테스트셋 별도 전처리
test_inputs['지습'] = test_inputs['지습'].fillna( test_inputs['지습'].mean() )
test_inputs['강우감지'] = test_inputs['강우감지'].fillna(value=0)
test_inputs['외부풍속'] = test_inputs['외부풍속'].fillna( test_inputs['외부풍속'].mean() )

test_input_sc = input_scaler.transform(test_inputs.iloc[:,3:].to_numpy())

test_input_ts = []
for i in output_sample['Sample_no']:
    # print(i)
    sample = test_input_sc[test_inputs['Sample_no'] == i]
    if len(sample) < 7:
        sample = np.append(np.zeros((7-len(sample), sample.shape[-1])) + sample[0], sample, axis=0)
        # sample = np.append(np.zeros((7-len(sample), sample.shape[-1])), sample, axis=0)
        # print(sample)
    sample = np.expand_dims(sample, axis=0)
    test_input_ts.append(sample)
test_input_ts = np.concatenate(test_input_ts, axis=0)

prediction = model.predict(test_input_ts)

prediction = output_scaler.inverse_transform(prediction)
output_sample[['생장길이', '줄기직경', '개화군']] = prediction
output_sample['생장길이'] = [i if i>0 else 0 for i in output_sample['생장길이']]
output_sample['줄기직경'] = [i if i>0 else 0 for i in output_sample['줄기직경']]
output_sample['개화군'] = [i if i>0 else 0 for i in output_sample['개화군']]


# 제출할 추론 결과 저장
output_sample.to_csv('prediction.csv', index=False)


