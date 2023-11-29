using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using Random = System.Random;

namespace ANN
{
    public class DNN
    {
        private int inputLayerNodeCount;

        private int beforeAddedNodeCount;

        private List<int> layerCount;
        private List<Matrix<double>> weights; // r,c
        private List<Matrix<double>> bias; // 1,n

        private List<Matrix<double>> layers; // 1,n

        private List<Matrix<double>> delta; // r,c

        private double learning_rate;

        public DNN(int inputLayerNodeCount, double learning_rate = 0.01)
        {
            // �Ű�� ���� �κ� 
            layerCount = new List<int>();
            weights = new List<Matrix<double>>();
            bias = new List<Matrix<double>>();
            this.inputLayerNodeCount = inputLayerNodeCount;
            this.learning_rate = learning_rate;

        } 

        //�Ű�� �ʱ�ȭ He�ʱ�ȭ ����� �����
        public void DNNInit()
        {
            Random random = new Random();
            //��� ���̾���
            for (int i = 0; i <= layerCount.Count; i++)
            {
                //����ġ �ʱ�ȭ
                for(int row = 0; row < weights[i].RowCount; row++)
                {
                    for(int col = 0; col < weights[i].ColumnCount; col++)
                    {
                        weights[i][row,col] = random.NextDouble() * Math.Sqrt(2.0f / weights[i].RowCount);
                    }
                }
                Debug.Log(weights[i]);
            }
            //������ 0���� �̹̼�����
        }

        //���̾� �߰� ������
        public void AddLayer(int nodeCount)
        {
            //����ġ �߰�
            if (weights.Count == 0)
                weights.Add(Matrix<double>.Build.Dense(inputLayerNodeCount, nodeCount, 0));
            else
                weights.Add(Matrix<double>.Build.Dense(beforeAddedNodeCount, nodeCount, 0));
            //�����߰�
            bias.Add(Matrix<double>.Build.Dense(1,nodeCount));

            layerCount.Add(nodeCount);
            beforeAddedNodeCount = nodeCount;
        }

        //�ƿ�ǲ ���̾� ���� AddLayer�� �����Ѵ��� ������ �����ؾ���
        //�ƿ�ǲ ���̾�� ������ ������������ ���� ���̾�ī��Ʈ�� �߰������ʴ´�
        public void SetOutputLayer(int outputLayerNode)
        {
            weights.Add(Matrix<double>.Build.Dense(beforeAddedNodeCount, outputLayerNode, 0));
        }

        //1,n ����
        public Matrix<double> Forward(Matrix<double> inputs)
        {
            if (inputs.ColumnCount != inputLayerNodeCount)
                Debug.LogError("��ǲ��尳���� ��ǲ�̴ٸ�");
            layers = new List<Matrix<double>>() { inputs };
            //��緹�̾ �ϳ��� �����Ѵ�.
            for (int i = 0; i < layerCount.Count; i++)
            {
                // value = ��ǲ * ����ġ + ���� 
                Matrix<double> values = ((inputs * weights[i]) + bias[i]);
                //���� dot product �� �ؼ� ����ġ�� ��ǲ��, ������ layer�� �߰��Ѵ�
                //������ ���̾��� ���� Ȱ��ȭ �Լ��� ��ħ
                //if(layerCount.Count - 1 != i) values = values.Map(x => (double)Math.Max(0,x));

                layers.Add(values.Map(x=> Math.Max(0,x)));
                //���� ���̾��� ��ǲ���� ���� ���̾��� �ƿ�ǲ�� ����
                inputs = values; 
            }

            // ������ ����� ���̾�� Ȱ��ȭ �Լ��� ������� �ʴ´�
            layers.Add(inputs * weights[layerCount.Count]);
            return layers.Last();
        }

        //�ַ� MSE�� �̿��Ͽ� ���� ������ ������ �������Ѵ�.
        //���� ��ġ �ȿ��� ������ �����Ҷ� ����� �̿��Ѵ� ��� ��ġ�� ������ ��ġ����ŭ ������ ��
        //������
        //errorMatrix�� ���� ��Ʈ���� [[0],[0],[�����Ѱ��ǿ���],[0]] �̷��� �������
        //delta���� ��� ����
        public void Backword(int selectedOutput, double expectedOutput)
        {
            //layers.Last().
            Matrix<double> errorMatrix;
            //������ ���̾� ��������� ������� deltaũ�� �ʱ�ȭ
            Matrix<double> deltaElement = Matrix<double>.Build.Dense(layers.Last().RowCount,layers.Last().ColumnCount,0);
            //������� Ȱ��ȭ�Լ��� ��� ���� ó���ϰ� DQNƯ���� �Ѱ����� �����ϱ⶧���� �Ѱ��� ������ ������
            // MSE���� �� ����� �������� �־��� �����
            deltaElement[0,selectedOutput] = (layers.Last()[0,selectedOutput] - expectedOutput);
            delta[layers.Count-2] = deltaElement;

            Debug.Log(deltaElement);

            //���̾��� ���� == ��Ÿ�� ����
            for (int i = layers.Count-2; i > 0; i--)
            {
                Debug.Log(weights.Count +"   "+ i);
                errorMatrix = deltaElement * weights[i].Transpose();
                deltaElement = Matrix<double>.Build.Dense(layers[i].RowCount,layers[i].ColumnCount,0);
                deltaElement = errorMatrix.PointwiseMultiply(layers[i].Map(x => x < 0 ? 0.0 : 1.0)); // ReLu�Լ��̺� 


                delta[i-1] = deltaElement;
                Debug.Log(deltaElement);
            }
            //���ٰ� �ջ� �Ϸ�
        }

        //batchsize�������� �װ� ����ŭ ����� ��
        public void updateDNN(int batchSize = 1)
        {
            for(int i = 0; i < layers.Count - 1; i++)
            {
                //��� * �н���
                 Debug.Log((layers[i].Transpose() * delta[i]) / batchSize * learning_rate);
                weights[i] += ((layers[i].Transpose() * delta[i]) / batchSize) * learning_rate;

                if (layers.Count - 2 == i) break;
                bias[i] += (delta[i] / batchSize) * learning_rate;
                 Debug.Log(delta[i] / batchSize * learning_rate);
            }
        }

        public void resetDelta()
        {
            delta = new List<Matrix<double>>();
            //ù��°�Ŵ� ��ǲ ������� �ǹ̾���
            /**
                Debug.Log(weights[0]);
                Debug.Log(bias[0]);
                Debug.Log(weights[weights.Count-1]);
                Debug.Log(bias[bias.Count -1]);
            **/
            for(int i = 1; i < layers.Count;i++)
            {
                delta.Add(Matrix<double>.Build.Dense(layers[i].RowCount, layers[i].ColumnCount, 0));
                //Debug.Log(layers[i]);
            }
        }
    }
}
