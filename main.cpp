#include "../GL/glwrapper.h"
#include "ImGuizmo/ImGuizmo.h"
//#include "utils.h"
#include <memory>
#include <sophus/se3.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <sophus/se3.hpp>

#include "costfun.h"
#include "struct.h"
#include "utils.h"

#include "tbb/tbb.h"

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/registration/ndt.h>


#include <pcl/kdtree/kdtree_flann.h>

#include <boost/program_options.hpp>
#include <thread>
class UnaryFactor: public gtsam::NoiseModelFactor1<gtsam::Pose3> {
    double mx_, my_; ///< X and Y measurements

public:
    UnaryFactor(gtsam::Key j, double x, double y, const gtsam::SharedNoiseModel& model):
        gtsam::NoiseModelFactor1<gtsam::Pose3>(model, j), mx_(x), my_(y) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& q,
                         boost::optional<gtsam::Matrix&> H = boost::none) const
    {
        if (H) (*H) = (gtsam::Matrix(2,6)<< 0,0,0,1,0,0, 0,0,0,0,1,0  ).finished();
        return (gtsam::Vector(2) << q.x() - mx_, q.y() - my_).finished();
    }
};

glm::vec2 clicked_point;
glm::vec3 view_translation{ 0,0,-30 };
float rot_x =0.0f;
float rot_y =0.0f;


void cursor_calback(GLFWwindow* window, double xpos, double ypos)
{
    ImGuiIO& io = ImGui::GetIO();
    if(!io.WantCaptureMouse) {
        const glm::vec2 p{-xpos, ypos};
        const auto d = clicked_point - p;
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS) {
            rot_x += 0.01 * d[1];
            rot_y += 0.01 * d[0];
        }
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_PRESS) {
            view_translation[2] += 0.02 * d[1];
        }
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_3) == GLFW_PRESS) {
            view_translation[1] -= 0.01 * d[1];
            view_translation[0] -= 0.01 * d[0];
        }
        clicked_point = p;
    }
}


void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

std::vector<Eigen::Matrix4d> trajectory;
std::vector<std::unique_ptr<structs::KeyFrame>> keyframes;

struct icp_result{
    Eigen::Matrix4d increment;
    int keyframePrev;
    int keyframeNext;
};

void register_ndt(const std::vector<std::unique_ptr<structs::KeyFrame>>& keyframes,
                  std::vector<icp_result>& icp_results, float ndt_resolution,
                  float loop_distance)
{
    std::mutex mutex_lck;
    icp_results.clear();
    int ndt_results = 0;
    tbb::parallel_for(tbb::blocked_range<size_t>(1,keyframes.size()),[&](const tbb::blocked_range<size_t>& r) {
        for (long i=r.begin();i<r.end();++i) {
            int prev = i - 1;
            int current = i;
            Eigen::Matrix4d increment = keyframes[prev]->mat.inverse() * keyframes[current]->mat;
            pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;
            // Setting scale dependent NDT parameters
            // Setting minimum transformation difference for termination condition.
            ndt.setTransformationEpsilon(0.01);
            // Setting maximum step size for More-Thuente line search.
            ndt.setStepSize(0.1);
            //Setting Resolution of NDT grid structure (VoxelGridCovariance).
            ndt.setResolution(ndt_resolution);

            ndt.setMaximumIterations(100);

            ndt.setInputSource(keyframes[current]->cloud);
            ndt.setInputTarget(keyframes[prev]->cloud);

            pcl::PointCloud<pcl::PointXYZI> t;
            Eigen::Matrix4f increment_f = increment.cast<float>();
            ndt.align(t, increment_f);

            if (ndt.hasConverged()) {
                icp_result r;
                r.increment = ndt.getFinalTransformation().cast<double>();
                r.keyframePrev = prev;
                r.keyframeNext = current;
                std::lock_guard<std::mutex> lck(mutex_lck);
                icp_results.push_back(r);
                std::cout << "icp_results " << icp_results.size() <<std::endl;
            }
        }
//                std::cout << increment.cast<float>() - ndt.getFinalTransformation() << std::endl;
//                keyframes[1]->mat = keyframes[0]->mat * ndt.getFinalTransformation().cast<double>();
    });
    std::cout << "loop closing"<<std::endl;
    std::vector<std::pair<int,int>> candidates;
    tbb::parallel_for(tbb::blocked_range<size_t>(1,keyframes.size()),[&](const tbb::blocked_range<size_t>& r) {
        for (long i=r.begin();i<r.end();++i) {
            for (int j=0; j< keyframes.size(); j++)
            {
                int prev = i;
                int current = j;
                if (abs(i-j) < 5) continue;
                const auto k1 = keyframes[i]->mat.col(3).head<3>();
                const auto k2 = keyframes[j]->mat.col(3).head<3>();
                double distance = (k1-k2).norm();
                if (distance < loop_distance){

                    Eigen::Matrix4d increment = keyframes[prev]->mat.inverse() * keyframes[current]->mat;
                    pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;
                    // Setting scale dependent NDT parameters
                    // Setting minimum transformation difference for termination condition.
                    ndt.setTransformationEpsilon(0.01);
                    // Setting maximum step size for More-Thuente line search.
                    ndt.setStepSize(0.1);
                    //Setting Resolution of NDT grid structure (VoxelGridCovariance).
                    ndt.setResolution(ndt_resolution);

                    ndt.setMaximumIterations( 100);

                    ndt.setInputSource(keyframes[current]->cloud);
                    ndt.setInputTarget(keyframes[prev]->cloud);

                    pcl::PointCloud<pcl::PointXYZI> t;
                    Eigen::Matrix4f increment_f = increment.cast<float>();
                    ndt.align(t, increment_f);
                    if (ndt.hasConverged()) {
                        icp_result r;
                        r.increment = ndt.getFinalTransformation().cast<double>();
                        r.keyframePrev = prev;
                        r.keyframeNext = current;
                        std::lock_guard<std::mutex> lck(mutex_lck);
                        icp_results.push_back(r);
                        std::cout << "icp_results " << icp_results.size()  <<"(" << distance << ")" <<std::endl;
                    }
                }
            }
        }
    });
}
namespace po = boost::program_options;
int main(int argc, char **argv) {

    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")
            ("dataset", po::value<std::string>(), "dataset")
            ("skip", po::value<int>(), "skip")
            ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    std::vector<icp_result> icp_results;
    std::vector<icp_result> icp_loop_closing;

    std::thread processing_thread;
    GLFWwindow *window;
    const char *glsl_version = "#version 130";
    if (!glfwInit())
        return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    window = glfwCreateWindow(960, 540, "rgbd_demo", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetCursorPosCallback(window, cursor_calback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    glfwSwapInterval(1);
    if (glewInit() != GLEW_OK) { return -1; }

    GLCall(glClearColor(0.4, 0.4, 0.4, 1));

    Renderer renderer;

    VertexBufferLayout layout;
    layout.Push<float>(3);
    layout.Push<float>(3);

    VertexArray va;
    VertexBuffer vb(gl_primitives::coordinate_system_vertex.data(),
                    gl_primitives::coordinate_system_vertex.size() * sizeof(float));
    va.AddBuffer(vb, layout);
    IndexBuffer ib(gl_primitives::coordinate_system_indices.data(), gl_primitives::coordinate_system_indices.size());

    Shader shader(shader_simple_v, shader_simple_f);
    Shader shader_pc(shader_pc_intensity_v, shader_pc_intensity_f);

    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, false);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // load matrix
    std::vector<structs::edge> Edges;
    //const std::string dataset{"/media/michal/ext/garaz2/scans/*.pcd"};
    const std::string dataset{vm["dataset"].as<std::string>()};
    std::vector<std::string> fns_raw = my_utils::glob(dataset);
    std::vector<std::string> fns;
    for (int i =0; i < fns_raw.size(); i+=vm["skip"].as<int>())
    {
        fns.push_back(std::string(fns_raw[i].begin(),fns_raw[i].end()-4));
    }


    double off_x = 0;
    double off_y = 0;

    for (const auto fn : fns){
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
        pcl::io::loadPCDFile<pcl::PointXYZI>(fn+".pcd",*cloud);
        trajectory.push_back(my_utils::loadMat(fn+".txt"));
        if (off_x == 0)
        {
            off_x = trajectory.front()(0,3);
            off_y = trajectory.front()(1,3);
        }
        trajectory.back()(0,3) -= off_x;
        trajectory.back()(1,3) -= off_y;

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_subsample(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::ApproximateVoxelGrid<pcl::PointXYZI> approximate_voxel_filter;
        approximate_voxel_filter.setLeafSize (0.1,0.1,0.1);
        approximate_voxel_filter.setInputCloud (cloud);
        approximate_voxel_filter.filter (*cloud_subsample);

        keyframes.emplace_back(std::make_unique<structs::KeyFrame>(cloud_subsample,  trajectory.back()));
    }
    const auto initial_poses = trajectory;


    Eigen::Matrix4f imgizmo {Eigen::Matrix4f::Identity()};
    int im_edited_frame = 1;
    int im_edited_frame_old = -1;
    float im_ndt_res = 0.5;
    float im_loop = 1.5;
    float im_gui_odometry = 1;
    while (!glfwWindowShouldClose(window)) {
        GLCall(glEnable(GL_DEPTH_TEST));
        GLCall(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGuiIO& io = ImGui::GetIO();
        ImGuizmo::BeginFrame();
        ImGuizmo::Enable(true);
        ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);

        /// OpenGL drawing
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        glm::mat4 scale = glm::mat4(0.1f);
        glm::mat4 proj = glm::perspective(30.f, 1.0f*width/height, 0.1f, 1000.0f);
        glm::mat4 model_translate = glm::translate(glm::mat4(1.0f), view_translation);
        glm::mat4 model_rotation_1 = glm::rotate(model_translate, rot_x, glm::vec3(1.0f, 0.0f, 0.0f));
        glm::mat4 model_rotation_2 = glm::rotate(model_rotation_1, rot_y, glm::vec3(0.0f, 0.0f, 1.0f));
        glm::mat4 model_rotation_3 = glm::rotate(model_rotation_2, (float)(0.5f*M_PI), glm::vec3(-1.0f, 0.0f, 0.0f));

        if (im_edited_frame >=0 && im_edited_frame < keyframes.size() ){
            if (im_edited_frame!= im_edited_frame_old){
                imgizmo = keyframes[im_edited_frame]->mat.cast<float>();
            }
            ImGuizmo::AllowAxisFlip(false);
            ImGuizmo::Manipulate(&model_rotation_3[0][0], &proj[0][0], ImGuizmo::TRANSLATE|ImGuizmo::ROTATE, ImGuizmo::LOCAL, imgizmo.data(), NULL);

        }

        im_edited_frame_old = im_edited_frame;

        shader.Bind(); // bind shader to apply uniform
        shader.setUniformMat4f("u_MVP", proj * model_rotation_3);
        renderer.Draw(va, ib, shader, GL_LINES);
//        // draw reference frame
        for (const auto &k : keyframes) {
            glm::mat4 local;
            Eigen::Map<Eigen::Matrix4f> map_local(&local[0][0]);
            map_local = k->mat.cast<float>();
            shader.setUniformMat4f("u_MVP", proj * model_rotation_3 * local * scale);
            renderer.Draw(va, ib, shader, GL_LINES);
        }
        GLCall(glPointSize(1));
        for ( int i =0; i < keyframes.size(); i++) {
            const auto &k = keyframes[i];
            shader_pc.Bind();
            glm::mat4 local;
            Eigen::Map<Eigen::Matrix4f> map_local(&local[0][0]);
            map_local = k->mat.cast<float>();

            if (i == im_edited_frame ){
                shader_pc.setUniform4f("u_COLORPC", 1,1,1,1);
            }else{

                shader_pc.setUniform4f("u_COLORPC", 1,0,0,1);
            }
            shader_pc.setUniformMat4f("u_MVPPC", proj * model_rotation_3 * local );
            renderer.DrawArray(k->va, shader_pc, GL_POINTS, k->cloud->size());
        }

        ImGui::Begin("SLAM Demo");
        ImGui::InputInt("Edited_frame",&im_edited_frame);
        ImGui::SameLine();
        if(ImGui::Button("hide gizmo")){
            im_edited_frame = -1;
        }
        if(ImGui::Button("load")) {
            std::vector<Eigen::Matrix4d> m_trajectory;
            m_trajectory.resize(keyframes.size());
            my_utils::LoadState("state.json", m_trajectory);
            for (int i =0; i < keyframes.size(); i++)
            {
                keyframes[i]->mat = m_trajectory[i];
            }
        }
        ImGui::SameLine();
        if(ImGui::Button("save")) {
            std::vector<Eigen::Matrix4d> m_trajectory;
            m_trajectory.resize(keyframes.size());
            for (int i =0; i < keyframes.size(); i++)
            {
                m_trajectory[i] = keyframes[i]->mat;
            }
            my_utils::saveState("state.json", m_trajectory);
        }

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
        if(ImGui::Button("reset view")){
            rot_x =0.0f;
            rot_y =0.0f;
            view_translation = glm::vec3{ 0,0,-30 };
            view_translation = glm::vec3{ 0,0,-30 };
        }

        ImGui::SliderFloat("im_ndt_res", &im_ndt_res, 0.f, 1.f);
        ImGui::SliderFloat("im_loop", &im_loop, 0.f, 5.f);
        if (ImGui::Button("ndt laser odometry"))
        {
            register_ndt(keyframes, icp_results,im_ndt_res, im_loop);
            using namespace std;
            using namespace gtsam;
            NonlinearFactorGraph graph;

            auto priorModel = noiseModel::Diagonal::Variances(
                    (Vector(6) << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12).finished());

            graph.add(PriorFactor<Pose3>(0, Pose3(keyframes[0]->mat), priorModel));

            //apply imu
            for (int i =0; i < keyframes.size(); i++) {
                auto priorModel = noiseModel::Diagonal::Variances(
                        (Vector(6) <<1e-5, 1e-5, 1e10, 1e10, 1e10, 1e10).finished());

                graph.add(PriorFactor<Pose3>(i, Pose3(trajectory[i]), priorModel));
            }
            for (int i =1; i < keyframes.size(); i++){
                auto odometryNoise = noiseModel::Diagonal::Variances(
                        (Vector(6) << 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1).finished());
                Eigen::Matrix4d update = trajectory[i-1].inverse() * trajectory[i];
                graph.emplace_shared<BetweenFactor<Pose3> >(i-1, i, Pose3(orthogonize(update)), odometryNoise);
            }
            for (int i =0; i < icp_results.size();i++)
            {
                auto odometryNoise = noiseModel::Diagonal::Variances(
                        (Vector(6) << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6).finished());
                int prev = icp_results[i].keyframePrev;
                int next = icp_results[i].keyframeNext;

                Eigen::Matrix4d update =  icp_results[i].increment;
                graph.emplace_shared<BetweenFactor<Pose3> >(prev, next, Pose3(orthogonize(update)), odometryNoise);
            }
            Values initial;
            for (int i =0; i < keyframes.size(); i++){
                initial.insert(i, Pose3(orthogonize(keyframes[i]->mat)));
            }
            Values result = LevenbergMarquardtOptimizer(graph, initial).optimize();
            for (int i =0; i < keyframes.size(); i++){
                auto v =  result.at<Pose3>(i);
                keyframes[i]->mat =v.matrix();
            }
            for (int i =0; i < icp_results.size();i++)
            {
                auto odometryNoise = noiseModel::Diagonal::Variances(
                        (Vector(6) << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6).finished());
                int prev = icp_results[i].keyframePrev;
                int next = icp_results[i].keyframeNext;

                Eigen::Matrix4d update =  icp_results[i].increment;
                graph.emplace_shared<BetweenFactor<Pose3> >(prev, next, Pose3(orthogonize(update)), odometryNoise);
            }
        }

        if (ImGui::Button("gtsam-relax")){
            using namespace std;
            using namespace gtsam;
            NonlinearFactorGraph graph;

            auto priorModel = noiseModel::Diagonal::Variances(
                    (Vector(6) << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12).finished());

            graph.add(PriorFactor<Pose3>(0, Pose3(keyframes[0]->mat), priorModel));
            graph.add(PriorFactor<Pose3>(im_edited_frame, Pose3(keyframes[im_edited_frame]->mat), priorModel));

            for (int i =1; i < im_edited_frame+1; i++){
                auto odometryNoise = noiseModel::Diagonal::Variances(
                        (Vector(6) << 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1).finished());
                Eigen::Matrix4d update = trajectory[i-1].inverse() * trajectory[i];
                graph.emplace_shared<BetweenFactor<Pose3> >(i-1, i, Pose3(orthogonize(update)), odometryNoise);
            }
            graph.print("\nFactor Graph:\n");  // print
            Values initial;
            for (int i =0; i < im_edited_frame+1; i++){
                initial.insert(i, Pose3(orthogonize(keyframes[i]->mat)));
            }
            Values result = LevenbergMarquardtOptimizer(graph, initial).optimize();
            result.print("Final Result:\n");
            for (int i =0; i < im_edited_frame+1; i++){
                auto v =  result.at<Pose3>(i);
                keyframes[i]->mat =v.matrix();
            }
        }
        if (ImGuizmo::IsOver()) {
        //if (ImGui::Button("t")) {
            if (im_edited_frame >= 0 && im_edited_frame < keyframes.size()) {
                const Eigen::Matrix4d before_inv{keyframes[im_edited_frame]->mat.inverse()};
                keyframes[im_edited_frame]->mat = imgizmo.cast<double>();
                std::cout << keyframes[im_edited_frame]->mat * before_inv << std::endl;
                for (int i = im_edited_frame + 1; i < keyframes.size(); i++) {
                    keyframes[i]->mat = keyframes[im_edited_frame]->mat * before_inv * keyframes[i]->mat;
                }
            }
        //}
        }
        if (ImGui::Button("reset")) {
            for (int i = 0; i < keyframes.size(); i++) {
                keyframes[i]->mat = trajectory[i].matrix();
            }
        }
        if (ImGui::Button("export")) {
            pcl::PointCloud<pcl::PointXYZI> result;
            for (int i = 0; i < keyframes.size(); i++) {
                pcl::PointCloud<pcl::PointXYZI> partial;
                const auto &mat  = keyframes[i]->mat;
                pcl::transformPointCloud(*(keyframes[i]->cloud), partial, mat.cast<float>());
                result += partial;
            }
            pcl::io::savePCDFileBinary("/tmp/cloud.pcd", result);
        }

        if (ImGui::Button("multiview ICP")) {
            std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> transformed;
            std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> non_transformed;
            std::vector<pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr> transformed_kdtree;

            ceres::Problem problem;
            std::vector<Sophus::SE3d> se3params;
            se3params.resize(keyframes.size());
            for (int i = 0; i < keyframes.size(); i++) {
                se3params[i]=Sophus::SE3d(Sophus::SE3d::fitToSE3(keyframes[i]->mat));
                problem.AddParameterBlock(se3params[i].data(), Sophus::SE3d::num_parameters,
                                          new LocalParameterizationSE3());
            }
            for (int i = 0; i < keyframes.size(); i++) {
                const auto &mat  = keyframes[i]->mat;

                pcl::PointCloud<pcl::PointXYZI>::Ptr subsample(new pcl::PointCloud<pcl::PointXYZI>());
                pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_subsample(new pcl::PointCloud<pcl::PointXYZI>());

                pcl::ApproximateVoxelGrid<pcl::PointXYZI> approximate_voxel_filter;
                approximate_voxel_filter.setLeafSize (0.25,0.25,0.25);
                approximate_voxel_filter.setInputCloud (keyframes[i]->cloud);
                approximate_voxel_filter.filter (*subsample);
                non_transformed.push_back(subsample);

                pcl::transformPointCloud(*subsample, *transformed_subsample, mat.cast<float>());
                transformed.push_back(transformed_subsample);

                pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr partial_subsample_kdtree(new pcl::KdTreeFLANN<pcl::PointXYZI>());
                partial_subsample_kdtree->setInputCloud(transformed_subsample);
                transformed_kdtree.push_back(partial_subsample_kdtree);
            }
            std::cout << "kdtrees done" <<std::endl;
            std::vector<std::pair<int,int>> pairs;
            for (int i=0; i < transformed.size(); i++)
            {
                for (int j=0; j < transformed.size(); j++)
                {
                    if (i==j) continue;
                    //if (abs(i-j)<5) continue;
                    auto pt1 =  Eigen::Affine3d(keyframes[i]->mat).translation();
                    auto pt2 =  Eigen::Affine3d(keyframes[j]->mat).translation();
                    double d = (pt1-pt2).norm();
                    if (d<5.0) {
                        pairs.push_back(std::pair<int, int>(i, j));
                    }
                }
            }
            for (auto pair : pairs)
            {
                const auto pp = transformed[pair.first];
                const auto pk = transformed_kdtree[pair.second];
                for (int p1_index = 0; p1_index < pp->size(); p1_index++ )
                {
                    pcl::PointXYZI pt1 = pp->at(p1_index);
                    std::vector<int> pointIdxRadiusSearch;
                    std::vector<float> pointRadiusSquaredDistance;
                    if (pk->radiusSearch(pt1, 0.5, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) {
                        int p2_index = pointIdxRadiusSearch[0];
                        Eigen::Vector4f p1 = non_transformed[pair.first]->at(p1_index).getVector4fMap();
                        Eigen::Vector4f p2 = non_transformed[pair.second]->at(p2_index).getVector4fMap();
                        ceres::LossFunction *loss = nullptr;//new ceres::CauchyLoss(0.2);
                        ceres::CostFunction *cost_function =costFunICP::Create(p1, p2);
                        problem.AddResidualBlock(cost_function, loss, se3params[pair.first].data(),se3params[pair.second].data());
                    }
                }
            }
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            options.minimizer_progress_to_stdout = true;
            options.max_num_iterations = 50;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            std::cout << summary.FullReport() << "\n";

            for (int i = 0; i < keyframes.size(); i++) {
                keyframes[i]->mat = se3params[i].matrix();
            }

        }
        ImGui::End();

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ImGui::Render();

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
        glfwPollEvents();

    }
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;

}