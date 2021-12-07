#pragma once
#include "../GL/glwrapper.h"
namespace structs{
    std::shared_ptr<float[]> pclToBuffer(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, int &len, float scale)
    {
        const int stride = 4;
        len = stride*cloud->size();
        std::shared_ptr<float[]> data (new float[len]);
        for (int i = 0; i < cloud->size();i++){
            data[stride*i+0] = scale*(*cloud)[i].x;
            data[stride*i+1] = scale*(*cloud)[i].y;
            data[stride*i+2] = scale*(*cloud)[i].z;
            data[stride*i+3] = (*cloud)[i].intensity;
        }
        return data;
    }


    struct edge{
        Eigen::Matrix4d se3;
        int id1;
        int id2;
    };
    struct KeyFrame{

        KeyFrame(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, Eigen::Matrix4d mat):
                cloud(cloud), mat(mat),data(pclToBuffer(cloud, len, 1.0f)),
                vb(data.get(), len* sizeof(float)),
                va()
        {
            timestamp = data[4];
            VertexBufferLayout layoutPc;
            layoutPc.Push<float>(3);
            layoutPc.Push<float>(1); // intensity
            va.AddBuffer(vb, layoutPc);
        }


        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
        Eigen::Matrix4d mat;
        double timestamp;
        const std::shared_ptr<float[]> data;
        int len;
        VertexBuffer vb;
        VertexArray va;
    };

}