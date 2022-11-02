#pragma once
#include "../GL/glwrapper.h"
#include <pcl/filters/approximate_voxel_grid.h>
#include <random>

namespace structs{
    std::shared_ptr<float[]> pclToBuffer(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, int &len, float scale)
    {
        const int stride = 4;
        len = stride*cloud->size();
        std::shared_ptr<float[]> data (new float[len]);
        for (int i = 0; i < cloud->size();i++){
            data[stride*i+0] = scale*(*cloud)[i].x;
            data[stride*i+1] = scale*(*cloud)[i].y;
            data[stride*i+2] = scale*(*cloud)[i].z;
            data[stride*i+3] = 1.0*(*cloud)[i].g/255;
        }
        return data;
    }


    struct KeyFrame{

        void subsample(float leaf_size){
            cloud_subsample = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
            pcl::ApproximateVoxelGrid<pcl::PointXYZRGB> approximate_voxel_filter;
            approximate_voxel_filter.setLeafSize (leaf_size,leaf_size,leaf_size);
            approximate_voxel_filter.setInputCloud (cloud);
            approximate_voxel_filter.filter (*cloud_subsample);
        }
        KeyFrame(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, Eigen::Matrix4d mat):
                cloud(cloud), mat(mat),data(pclToBuffer(cloud, len, 1.0f)),
                vb(data.get(), len* sizeof(float)),
                va()
        {
            timestamp = data[4];
            VertexBufferLayout layoutPc;
            layoutPc.Push<float>(3);
            layoutPc.Push<float>(1); // intensity
            va.AddBuffer(vb, layoutPc);
            //subsample(0.1);

            std::random_device rd;  // Will be used to obtain a seed for the random number engine
            std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
            std::uniform_real_distribution<float> dis(0, 1.0);
            color = Eigen::Vector3f{dis(gen),dis(gen),dis(gen)};
        }


        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_subsample;

        Eigen::Matrix4d mat;
        double timestamp;
        const std::shared_ptr<float[]> data;
        int len;
        VertexBuffer vb;
        VertexArray va;
        Eigen::Vector3f color;
        Eigen::Vector2d UTM;
        Eigen::Vector2d UTM_offset;
        bool gnss_valid{false};
        Eigen::Vector4d groundDir;
        std::string fn;

    };

}