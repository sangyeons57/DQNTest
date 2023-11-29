using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting.Antlr3.Runtime.Tree;
using UnityEngine;
using UnityEngine.UI;
using Random = System.Random;

namespace ANN
{
    public class DNN
    {
        //아웃풋 레이어 설정했는지
        private bool isFinishAddLayer;

        private int inputLayerNodeCount;
        private int beforeAddedNodeCount;

        private int batchSize;

        //입력층 은닉층 출력층의 노드개수
        private List<int> layerCount;
        private List<Matrix<double>> weights; // r,c
        private List<Matrix<double>> bias; // 1,n

        //순전파때 저장값
        private List<Matrix<double>> layers; // 1,n

        //역전파떄 오차값의 합
        private List<Matrix<double>> delta; // r,c

        private double learning_rate;

        public DNN(int inputLayerNodeCount, double learning_rate = 0.01)
        {
            // 신경망 생성 부분 
            layerCount = new List<int>();
            weights = new List<Matrix<double>>();
            bias = new List<Matrix<double>>();
            this.inputLayerNodeCount = inputLayerNodeCount;
            this.learning_rate = learning_rate;
        } 

        //신경망 초기화 He초기화 방식을 사용함
        public void DNNInit()
        {
            Random random = new Random();
            //모든 레이어의
            for (int i = 0; i < layerCount.Count; i++)
            {
                //가중치 초기화
                for(int row = 0; row < weights[i].RowCount; row++)
                {
                    for(int col = 0; col < weights[i].ColumnCount; col++)
                    {
                        weights[i][row,col] = random.NextDouble() * Math.Sqrt(2.0f / weights[i].RowCount);
                    }
                }
            }
            //편향은 0으로 이미설정됨
        }

        //레이어 추가 은닉층
        public void AddLayer(int nodeCount, bool isOutputLayer = false)
        {
            if (isFinishAddLayer) return;

            //가중치 추가
            if (weights.Count == 0)
                weights.Add(Matrix<double>.Build.Dense(inputLayerNodeCount, nodeCount, 0));
            else
                weights.Add(Matrix<double>.Build.Dense(beforeAddedNodeCount, nodeCount, 0));
            if(isOutputLayer)
                isFinishAddLayer = true;
            else //편향추가
                bias.Add(Matrix<double>.Build.Dense(1,nodeCount));

            layerCount.Add(nodeCount);
            beforeAddedNodeCount = nodeCount;
        }

        //1,n 형식
        public Matrix<double> Forward(Matrix<double> inputs)
        {
            if (inputs.ColumnCount != inputLayerNodeCount) Debug.LogError("인풋노드개수랑 인풋이다름");
            if (!isFinishAddLayer) Debug.LogError("아웃풋 레이어가 세팅이 안됨");
            layers = new List<Matrix<double>>() { inputs };
            //모든레이어를 하나씩 진행한다.
            for (int i = 0; i < layerCount.Count - 1; i++)
            {
                // value = 인풋 * 가중치 + 편향 
                //내적 dot product 을 해서 가중치와 인풋을, 값들을 layer에 추가한다
                Matrix<double> values = ReLU((inputs * weights[i]) + bias[i]);
                layers.Add(values);
                //다음 레이어의 인풋으로 현제 레이어의 아웃풋을 넣음
                inputs = values;
            }
            //마지막 레이어빼고는 전부 활성화 함수를 거침, 편향도 없음
            layers.Add(inputs * weights[layerCount.Count - 1]);

            return layers.Last();
        }

        //주로 MSE를 이용하여 들어온 오차를 가지고 역전파한다.
        //또한 배치 안에서 오차를 적용할때 평균을 이용한다 모든 배치를 적용후 배치수만큼 나누면 됨
        //역전파
        //errorMatrix는 오차 메트릭스 [[0],[0],[선택한값의오차],[0]] 이렇게 들어오며됨
        //delta값은 계속 합함
        public void Backword(Matrix<double> realOutput)
        {
            //layers.Last().
            Matrix<double> errorMatrix = (realOutput - layers.Last());
            //출력층은 편향이없고 활성화함수가없음
            Matrix<double> deltaElement = errorMatrix.PointwiseMultiply(layers.Last());
            //오차값저장 사이즈-1 ,입력층 -1 = layer.count -2
            delta[layers.Count - 2] += deltaElement;

            //다음 에러로 변경
            errorMatrix = deltaElement * weights[layers.Count - 2].Transpose();

            //출력층은 활성화함수가 없어서 따로 처리하고 DQN특성상 한가지만 전파하기때문에 한가지 오차만 만들음
            // MSE값을 각 노드의 오차값에 넣어준 모습임
            //레이어의 개수 == 델타의 개수
            for (int i = layers.Count-2; i > 0; i--)
            {
                deltaElement = errorMatrix.PointwiseMultiply(ReLUDerivative(layers[i])); // ReLu함수미분 
                errorMatrix = deltaElement * weights[i-1].Transpose();

                delta[i-1] += deltaElement;
            }
            //델다값 합산 완료
            //역전파 한번당 배치사이즈 증가
            batchSize++;
        }

        //batchsize가들어오면 그거 수만큼 평균을 냄
        public void updateDNN()
        {
            // 그리고 layer.count - 2은 마지막 은닉층,
            // 0번째는 입력층layer.count - 1은 출력층
            for(int i = 0; i < layers.Count - 1; i++)
            {
                //평균 * 학습률
                //(전 레이어의 입력 (내적) 오차값) / 배치 크기 = 각 가중치의 평균
                Debug.Log(((layers[i].Transpose() * delta[i]) / batchSize) * learning_rate);
                weights[i] += ((layers[i].Transpose() * delta[i]) / batchSize) * learning_rate;

                // 편향의 layer.count - 2  는 출력층 하지만 출력층은 편향이 없음으로 건너뜀
                if (i == layers.Count - 2) break;

                Debug.Log( (delta[i] / batchSize) * learning_rate);
                bias[i] += (delta[i] / batchSize) * learning_rate;
            }
        }

        public void resetDelta()
        {
            delta = new List<Matrix<double>>();
            batchSize = 0;
            for(int i = 1; i < layers.Count;i++)
            {
                delta.Add(Matrix<double>.Build.Dense(layers[i].RowCount, layers[i].ColumnCount, 0));
            }
        }

        static Matrix<double> ReLU(Matrix<double> matrix)
        {
            return matrix.Map(x => x > 0 ? x : 0.01 * x);
        }

        static Matrix<double> ReLUDerivative(Matrix<double> matrix)
        {
            return matrix.Map(x => x > 0 ? 1.0 : 0.01);
        }
    }
}
