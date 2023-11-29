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
        //�ƿ�ǲ ���̾� �����ߴ���
        private bool isFinishAddLayer;

        private int inputLayerNodeCount;
        private int beforeAddedNodeCount;

        private int batchSize;

        //�Է��� ������ ������� ��尳��
        private List<int> layerCount;
        private List<Matrix<double>> weights; // r,c
        private List<Matrix<double>> bias; // 1,n

        //�����Ķ� ���尪
        private List<Matrix<double>> layers; // 1,n

        //�����ċ� �������� ��
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
            for (int i = 0; i < layerCount.Count; i++)
            {
                //����ġ �ʱ�ȭ
                for(int row = 0; row < weights[i].RowCount; row++)
                {
                    for(int col = 0; col < weights[i].ColumnCount; col++)
                    {
                        weights[i][row,col] = random.NextDouble() * Math.Sqrt(2.0f / weights[i].RowCount);
                    }
                }
            }
            //������ 0���� �̹̼�����
        }

        //���̾� �߰� ������
        public void AddLayer(int nodeCount, bool isOutputLayer = false)
        {
            if (isFinishAddLayer) return;

            //����ġ �߰�
            if (weights.Count == 0)
                weights.Add(Matrix<double>.Build.Dense(inputLayerNodeCount, nodeCount, 0));
            else
                weights.Add(Matrix<double>.Build.Dense(beforeAddedNodeCount, nodeCount, 0));
            if(isOutputLayer)
                isFinishAddLayer = true;
            else //�����߰�
                bias.Add(Matrix<double>.Build.Dense(1,nodeCount));

            layerCount.Add(nodeCount);
            beforeAddedNodeCount = nodeCount;
        }

        //1,n ����
        public Matrix<double> Forward(Matrix<double> inputs)
        {
            if (inputs.ColumnCount != inputLayerNodeCount) Debug.LogError("��ǲ��尳���� ��ǲ�̴ٸ�");
            if (!isFinishAddLayer) Debug.LogError("�ƿ�ǲ ���̾ ������ �ȵ�");
            layers = new List<Matrix<double>>() { inputs };
            //��緹�̾ �ϳ��� �����Ѵ�.
            for (int i = 0; i < layerCount.Count - 1; i++)
            {
                // value = ��ǲ * ����ġ + ���� 
                //���� dot product �� �ؼ� ����ġ�� ��ǲ��, ������ layer�� �߰��Ѵ�
                Matrix<double> values = ReLU((inputs * weights[i]) + bias[i]);
                layers.Add(values);
                //���� ���̾��� ��ǲ���� ���� ���̾��� �ƿ�ǲ�� ����
                inputs = values;
            }
            //������ ���̾��� ���� Ȱ��ȭ �Լ��� ��ħ, ���⵵ ����
            layers.Add(inputs * weights[layerCount.Count - 1]);

            return layers.Last();
        }

        //�ַ� MSE�� �̿��Ͽ� ���� ������ ������ �������Ѵ�.
        //���� ��ġ �ȿ��� ������ �����Ҷ� ����� �̿��Ѵ� ��� ��ġ�� ������ ��ġ����ŭ ������ ��
        //������
        //errorMatrix�� ���� ��Ʈ���� [[0],[0],[�����Ѱ��ǿ���],[0]] �̷��� �������
        //delta���� ��� ����
        public void Backword(Matrix<double> realOutput)
        {
            //layers.Last().
            Matrix<double> errorMatrix = (realOutput - layers.Last());
            //������� �����̾��� Ȱ��ȭ�Լ�������
            Matrix<double> deltaElement = errorMatrix.PointwiseMultiply(layers.Last());
            //���������� ������-1 ,�Է��� -1 = layer.count -2
            delta[layers.Count - 2] += deltaElement;

            //���� ������ ����
            errorMatrix = deltaElement * weights[layers.Count - 2].Transpose();

            //������� Ȱ��ȭ�Լ��� ��� ���� ó���ϰ� DQNƯ���� �Ѱ����� �����ϱ⶧���� �Ѱ��� ������ ������
            // MSE���� �� ����� �������� �־��� �����
            //���̾��� ���� == ��Ÿ�� ����
            for (int i = layers.Count-2; i > 0; i--)
            {
                deltaElement = errorMatrix.PointwiseMultiply(ReLUDerivative(layers[i])); // ReLu�Լ��̺� 
                errorMatrix = deltaElement * weights[i-1].Transpose();

                delta[i-1] += deltaElement;
            }
            //���ٰ� �ջ� �Ϸ�
            //������ �ѹ��� ��ġ������ ����
            batchSize++;
        }

        //batchsize�������� �װ� ����ŭ ����� ��
        public void updateDNN()
        {
            // �׸��� layer.count - 2�� ������ ������,
            // 0��°�� �Է���layer.count - 1�� �����
            for(int i = 0; i < layers.Count - 1; i++)
            {
                //��� * �н���
                //(�� ���̾��� �Է� (����) ������) / ��ġ ũ�� = �� ����ġ�� ���
                Debug.Log(((layers[i].Transpose() * delta[i]) / batchSize) * learning_rate);
                weights[i] += ((layers[i].Transpose() * delta[i]) / batchSize) * learning_rate;

                // ������ layer.count - 2  �� ����� ������ ������� ������ �������� �ǳʶ�
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
