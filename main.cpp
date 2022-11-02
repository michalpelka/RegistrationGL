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
#include <gtsam/navigation/GPSFactor.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/filter.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <boost/program_options.hpp>
#include <thread>
glm::vec2 clicked_point;
glm::vec3 view_translation{ 0,0,-30 };
float rot_x =M_PI/2;
float rot_y =0.0f;
std::vector<Eigen::Matrix4d> trajectory;
std::vector<Eigen::Matrix4d> trajectory_noskip;
std::vector<double> trajectory_ts;
std::vector<double> trajectory_ts_noskip;

std::vector<Eigen::Matrix4d> trajectory_interpolated;
std::vector<double> trajectory_ts_interpolated;


std::vector<std::unique_ptr<structs::KeyFrame>> keyframes;
std::unique_ptr<structs::KeyFrame> gt_keyframe{nullptr};

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
            view_translation[1] -= 0.05 * d[1];
            view_translation[0] -= 0.05 * d[0];
        }
        clicked_point = p;
    }
}


void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}


struct icp_result{
    Eigen::Matrix4d increment;
    int keyframePrev;
    int keyframeNext;
};

void register_ndt(const std::vector<std::unique_ptr<structs::KeyFrame>>& keyframes,
                  std::vector<icp_result>& icp_results, float ndt_resolution,
                  float loop_distance, float ndt_res_loop, int frames_odom)
{
    std::mutex mutex_lck;
    icp_results.clear();
    int ndt_results = 0;
    tbb::parallel_for(tbb::blocked_range<size_t>(1,keyframes.size()),[&](const tbb::blocked_range<size_t>& r) {
        for (long i=r.begin();i<r.end();++i) {
            //int prev = i - 1;
            int current = i;
            if (frames_odom<=0) continue;
            for (int prev = i-frames_odom ; prev<i+frames_odom; prev++) {
                if (prev>=keyframes.size())continue;
                if (prev<0)continue;
                if (prev==i)continue;
                Eigen::Matrix4d increment = keyframes[prev]->mat.inverse() * keyframes[current]->mat;
                pcl::NormalDistributionsTransform<pcl::PointXYZRGB, pcl::PointXYZRGB> ndt;
                // Setting scale dependent NDT parameters
                // Setting minimum transformation difference for termination condition.
                ndt.setTransformationEpsilon(0.01);
                // Setting maximum step size for More-Thuente line search.
                ndt.setStepSize(0.1);
                //Setting Resolution of NDT grid structure (VoxelGridCovariance).
                ndt.setResolution(ndt_resolution);

                ndt.setMaximumIterations(50);

                ndt.setInputSource(keyframes[current]->cloud);
                ndt.setInputTarget(keyframes[prev]->cloud);

                pcl::PointCloud<pcl::PointXYZRGB> t;
                Eigen::Matrix4f increment_f = increment.cast<float>();
                ndt.align(t, increment_f);

                if (ndt.hasConverged()) {
                    icp_result r;
                    r.increment = ndt.getFinalTransformation().cast<double>();
                    r.keyframePrev = prev;
                    r.keyframeNext = current;
                    std::lock_guard<std::mutex> lck(mutex_lck);
                    icp_results.push_back(r);
                    std::cout << "icp_results " << icp_results.size() << " " << prev << " -> " << current << std::endl;
                }
            }
        }
//                std::cout << increment.cast<float>() - ndt.getFinalTransformation() << std::endl;
//                keyframes[1]->mat = keyframes[0]->mat * ndt.getFinalTransformation().cast<double>();
    });
    std::cout << "loop closing"<<std::endl;
    std::vector<std::pair<int,int>> candidates;
    std::mutex mtx;
    tbb::parallel_for(tbb::blocked_range<size_t>(0,keyframes.size()),[&](const tbb::blocked_range<size_t>& r) {
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
                    pcl::NormalDistributionsTransform<pcl::PointXYZRGB, pcl::PointXYZRGB> ndt;
                    // Setting scale dependent NDT parameters
                    // Setting minimum transformation difference for termination condition.
                    ndt.setTransformationEpsilon(0.01);
                    // Setting maximum step size for More-Thuente line search.
                    ndt.setStepSize(0.1);
                    //Setting Resolution of NDT grid structure (VoxelGridCovariance).
                    ndt.setResolution(ndt_res_loop);

                    ndt.setMaximumIterations( 200);

                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr current_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr prev_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

                    pcl::ApproximateVoxelGrid<pcl::PointXYZRGB> approximate_voxel_filter;
                    approximate_voxel_filter.setLeafSize (ndt_res_loop/2,ndt_res_loop/2,ndt_res_loop/2);
                    approximate_voxel_filter.setInputCloud (keyframes[current]->cloud);
                    approximate_voxel_filter.filter (*current_ptr);


                    approximate_voxel_filter.setInputCloud (keyframes[prev]->cloud);
                    approximate_voxel_filter.filter (*prev_ptr);

                    ndt.setInputSource(current_ptr);
                    ndt.setInputTarget(prev_ptr);

                    pcl::PointCloud<pcl::PointXYZRGB> t;
                    Eigen::Matrix4f increment_f = increment.cast<float>();
                    ndt.align(t, increment_f);

                    if (ndt.hasConverged()) {
                        icp_result r;
                        r.increment = ndt.getFinalTransformation().cast<double>();
                        r.keyframePrev = prev;
                        r.keyframeNext = current;
                        std::lock_guard<std::mutex> lck(mutex_lck);
                        icp_results.push_back(r);
                        std::cout << "icp_results " << icp_results.size()  <<"(" << distance << ")" << prev <<" -> " << current <<std::endl;
                    }
                }
            }
        }
    });
    std::vector<std::pair<int,int>> end_to_end {{0, keyframes.size()-1},{keyframes.size()-1,1}};
    for (auto pair : end_to_end )
    {
        int prev = pair.first;
        int current = pair.second;
        const auto k1 = keyframes[prev]->mat.col(3).head<3>();
        const auto k2 = keyframes[current]->mat.col(3).head<3>();
        double distance = (k1-k2).norm();

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr current_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr prev_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

        pcl::ApproximateVoxelGrid<pcl::PointXYZRGB> approximate_voxel_filter;
        approximate_voxel_filter.setLeafSize (1,1,1);
        approximate_voxel_filter.setInputCloud (keyframes[current]->cloud);
        approximate_voxel_filter.filter (*current_ptr);


        approximate_voxel_filter.setInputCloud (keyframes[prev]->cloud);
        approximate_voxel_filter.filter (*prev_ptr);

        Eigen::Matrix4d increment = keyframes[prev]->mat.inverse() * keyframes[current]->mat;
        pcl::NormalDistributionsTransform<pcl::PointXYZRGB, pcl::PointXYZRGB> ndt;
        // Setting scale dependent NDT parameters
        // Setting minimum transformation difference for termination condition.
        ndt.setTransformationEpsilon(0.05);
        // Setting maximum step size for More-Thuente line search.
        ndt.setStepSize(0.1);
        //Setting Resolution of NDT grid structure (VoxelGridCovariance).
        ndt.setResolution(10);

        ndt.setMaximumIterations(200);

        ndt.setInputSource(current_ptr);
        ndt.setInputTarget(prev_ptr);

        pcl::PointCloud<pcl::PointXYZRGB> t;
        Eigen::Matrix4f increment_f = increment.cast<float>();
        ndt.align(t, increment_f);
        if (ndt.hasConverged()) {
            icp_result r;
            r.increment = ndt.getFinalTransformation().cast<double>();
            r.keyframePrev = prev;
            r.keyframeNext = current;
            std::lock_guard<std::mutex> lck(mutex_lck);
            icp_results.push_back(r);
            std::cout << "icp_results " << icp_results.size() << "(" << distance << ")" << prev << " -> " << current
                      << std::endl;
        }

    }

}
struct gt_overlay{

    const std::vector<float> positions
            {
                    -1.f, -1.f, 0.0f, 0.0f,// bottom left
                     1.f, -1.f, 1.0f, 0.0f,// bottom right
                     1.f,  1.f, 1.0f, 1.0f,// top right
                    -1.f,  1.f, 0.0f, 1.0f,// top left
            };

    const std::vector<unsigned int> indicies {
            0,1,2, 2,3,0,
    };


    std::string fn;
    Texture texture;
    Eigen::Vector2d  real_size;
    Eigen::Vector2d  img_size;

    VertexBuffer vb;
    IndexBuffer ib;
    VertexArray va;
    Shader shader;
public:
    gt_overlay(const std::string& fn):fn(fn), texture(fn),
    vb(positions.data(),positions.size()* sizeof(float)),
    ib(indicies.data(), indicies.size()),
    va(),
    shader(shader_simple_tex_v,shader_simple_tex_f)
    {
        VertexBufferLayout layout;
        layout.Push<float>(2);
        layout.Push<float>(2);
        va.AddBuffer(vb,layout);
    }
    explicit operator bool() const {
        return fn.length();
    }

};

int main(int argc, char **argv) {
    std::shared_ptr<gt_overlay> gt_img_overlay;
//    Eigen::Affine3d laser_offset{Eigen::Affine3d::Identity()};
//    laser_offset.translation() = Eigen::Vector3d{0.2,0.0,0.5};
//    auto Q = Eigen::Quaterniond::Identity();//Eigen::Quaterniond{ 0.96593,0.0, 0.0, -0.25882};
//    Q.normalize();
//    laser_offset.rotate(Q);

    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")
            ("dataset", po::value<std::string>(), "dataset")
            ("gt", po::value<std::string>(), "ground truth")
            ("gt_plane", po::value<int>(), "gt_plane")
            ("gt_img", po::value<std::string>(), "ground truth image")
            ("gt_img_length", po::value<int>(), "ground truth image length")
            ("postfix", po::value<std::string>()->default_value(".pcd"), "postfix to pcd")
            ("skip", po::value<int>()->default_value(1), "skip")
            ("laser_offset", po::value<float>()->default_value(0.f), "laser_offset")
            ("json", po::value<std::string>(), "json state")
            ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    Eigen::Affine3d laser_offset{Eigen::Affine3d::Identity()};
    laser_offset.translation() = Eigen::Vector3d{0.2,0.0,0.5};
    const double angle = M_PI*(vm["laser_offset"].as<float>())/180.0;
    auto Q = Eigen::Quaterniond{ std::sin(angle/2.0),0.0, 0.0, std::cos(angle/2.0)};
    Q.normalize();
    laser_offset.rotate(Q);


    std::vector<icp_result> icp_results;
    std::vector<icp_result> icp_loop_closing;
    std::map<int,Eigen::Matrix4d> icp_gt_resutls;

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
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
    glfwSetCursorPosCallback(window, cursor_calback);
    const std::string json_config = (vm.count("json")>0)?(vm["json"].as<std::string>()):"config.json";
    std::cout << "json_config " << json_config << std::endl;
    // load matrix
    const std::string dataset{vm["dataset"].as<std::string>()};
    std::vector<std::string> fns_raw = my_utils::glob(dataset);
    std::vector<std::string> fns;
    std::vector<std::string> fns_noskip;
    const std::string postfix  = vm["postfix"].as<std::string>();
    for (int i =0; i < fns_raw.size(); i+=1)
    {
        fns_noskip.push_back(std::string(fns_raw[i].begin(),fns_raw[i].end()-postfix.size()));
    }
    for (int i =0; i < fns_raw.size(); i+=vm["skip"].as<int>())
    {
        fns.push_back(std::string(fns_raw[i].begin(),fns_raw[i].end()-postfix.size()));
    }

    double off_x = 0;
    double off_y = 0;

    for (const auto fn : fns){
        std::cout << "fn " << fn << std::endl;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::io::loadPCDFile<pcl::PointXYZRGB>(fn+postfix,*cloud);
        trajectory.push_back(my_utils::loadMat(fn+".txt"));
        Eigen::Vector2d utm = my_utils::loadNovatel(fn+".novatel");
        Eigen::Vector4d ground_dir = my_utils::loadGround(fn+"_ground.txt");
        double ts = 0;
        std::ifstream  ifn (fn+".ts");
        ifn >> ts;
        trajectory_ts.push_back(ts);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_subsample(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_subample_nan(new pcl::PointCloud<pcl::PointXYZRGB>());



        pcl::ApproximateVoxelGrid<pcl::PointXYZRGB> approximate_voxel_filter;
        approximate_voxel_filter.setLeafSize (0.5,0.5,0.5);
        approximate_voxel_filter.setInputCloud (cloud);
        approximate_voxel_filter.filter (*cloud_subsample);
        cloud_subample_nan->reserve(cloud->size());
        for (const auto & p : *cloud_subsample){
            Eigen::Vector3f pp = p.getVector3fMap();
            if (pp.norm()< 500 && pp.norm()>0)
            {
                cloud_subample_nan->push_back(p);
            }
        }
        keyframes.emplace_back(std::make_unique<structs::KeyFrame>(cloud_subample_nan,  trajectory.back()));
        keyframes.back()->UTM = utm;
        keyframes.back()->groundDir = ground_dir;
        keyframes.back()->fn = fn;
    }
    // update UTM;
    Eigen::Vector2d centroid {0.,0.};
    int valid_utm = 0;
    for (const auto& k: keyframes){
        if (k->UTM.x()>0 &&k->UTM.y()>0) {
            centroid += k->UTM;
            k->gnss_valid = true;
            valid_utm++;
        }
    }
    centroid = centroid / valid_utm;
    std::cout <<"UTM frames :" << std::endl;
    for (const auto& k: keyframes){
        if (k->gnss_valid) {
            k->UTM_offset = k->UTM - centroid;
            std::cout << k->UTM_offset.transpose();
            std::cout << std::endl;
        }
    }



    for (const auto fn : fns_noskip){
        trajectory_noskip.push_back(my_utils::loadMat(fn+".txt"));
        double ts = 0;
        std::ifstream  ifn (fn+".ts");
        ifn >> ts;
        trajectory_ts_noskip.push_back(ts);
        trajectory_noskip.back()(0,3) -= off_x;
        trajectory_noskip.back()(1,3) -= off_y;

    }

    const auto initial_poses = trajectory;

    // load texture
    if (vm.count("gt_img")>0){
        const std::string fn(vm["gt_img"].as<std::string>());
        gt_img_overlay = std::make_shared<gt_overlay>(fn);
    }
    
    if (vm.count("gt"))
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::io::loadPCDFile<pcl::PointXYZRGB>(vm["gt"].as<std::string>(),*cloud);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_subsample(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::ApproximateVoxelGrid<pcl::PointXYZRGB> approximate_voxel_filter;
        approximate_voxel_filter.setLeafSize (0.1,0.1,0.1);
        approximate_voxel_filter.setInputCloud (cloud);
        approximate_voxel_filter.filter (*cloud_subsample);
        gt_keyframe = std::make_unique<structs::KeyFrame>(cloud_subsample,  Eigen::Matrix4d::Identity());
    }
    if (vm.count("gt_plane"))
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        for (float f =-200; f < 200; f+=0.2)
        {
            for (float g =-200; g < 200; g+=0.2){
                pcl::PointXYZRGB p;
                p.getVector3fMap() = Eigen::Vector3f(f,g,0);
                cloud->push_back(p);
            }

        }
        gt_keyframe = std::make_unique<structs::KeyFrame>(cloud,  Eigen::Matrix4d::Identity());
    }


    Eigen::Matrix4f imgizmo {Eigen::Matrix4f::Identity()};
    int im_edited_frame = 1;
    int im_edited_frame_old = -1;
    int im_frames_odom=1;

    float im_ndt_res = 0.5;
    float im_ndt_res_loop = 2.5;
    float im_loop = 5.5;
    float im_gui_odometry = 1;
    bool im_draw_only_edited{false};
    bool im_top_ortho{false};
    float im_ortho_scale = 50;
    float im_ortho_height = -50;
    float im_ortho_slice = 100;


    bool im_run_ndt{false};


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
        glm::mat4 proj;
        glm::mat4 model_rotation_3;
        if (im_top_ortho) {
            proj = glm::ortho(-im_ortho_scale* width / height, im_ortho_scale * width / height, -im_ortho_scale, im_ortho_scale, im_ortho_slice,
                              im_ortho_slice+im_ortho_height);
            glm::mat4 model_translate = glm::translate(glm::mat4(1.0f), view_translation);
            glm::mat4 model_rotation_1 = glm::rotate(model_translate, float(M_PI/2), glm::vec3(1.0f, 0.0f, 0.0f));
            model_rotation_3 = glm::rotate(model_rotation_1, (float)(0.5f*M_PI), glm::vec3(-1.0f, 0.0f, 0.0f));

        }else{
            proj = glm::perspective(30.f, 1.0f*width/height, 0.1f, 1000.0f);
            glm::mat4 model_translate = glm::translate(glm::mat4(1.0f), view_translation);
            glm::mat4 model_rotation_1 = glm::rotate(model_translate, rot_x, glm::vec3(1.0f, 0.0f, 0.0f));
            glm::mat4 model_rotation_2 = glm::rotate(model_rotation_1, rot_y, glm::vec3(0.0f, 0.0f, 1.0f));
            model_rotation_3 = glm::rotate(model_rotation_2, (float)(0.5f*M_PI), glm::vec3(-1.0f, 0.0f, 0.0f));
        }


        if (im_edited_frame >=0 && im_edited_frame < keyframes.size() ){
            if (im_edited_frame!= im_edited_frame_old){
                imgizmo = keyframes[im_edited_frame]->mat.cast<float>();
            }
            ImGuizmo::AllowAxisFlip(false);
            ImGuizmo::Manipulate(&model_rotation_3[0][0], &proj[0][0], ImGuizmo::TRANSLATE|ImGuizmo::ROTATE_Z|ImGuizmo::ROTATE_X|ImGuizmo::ROTATE_Y, ImGuizmo::LOCAL, imgizmo.data(), NULL);
        }

        im_edited_frame_old = im_edited_frame;

        if (gt_img_overlay){
            gt_img_overlay->shader.Bind();
            gt_img_overlay->texture.Bind(1);
            glm::mat4 local;
            Eigen::Map<Eigen::Matrix4f> map_local(&local[0][0]);
            map_local = Eigen::Matrix4f::Identity();
            const float larger = std::min(gt_img_overlay->texture.GetWidth(),gt_img_overlay->texture.GetHeight());
            map_local(0,0) = static_cast<float>(gt_img_overlay->texture.GetWidth())*44.1/larger;
            map_local(1,1) = static_cast<float>(gt_img_overlay->texture.GetHeight())*44.1/larger;
            map_local(2,2) = 1;

            gt_img_overlay->shader.setUniformMat4f("u_MVP5", proj * model_rotation_3 * local );
            gt_img_overlay->shader.setUniform1i("u_Texture", 1);
            renderer.Draw(gt_img_overlay->va, gt_img_overlay->ib, gt_img_overlay->shader, GL_TRIANGLES);
            gt_img_overlay->texture.Unbind();
            gt_img_overlay->shader.Unbind();
        }

        shader.Bind(); // bind shader to apply uniform
        shader.setUniformMat4f("u_MVP", proj * model_rotation_3);
        renderer.Draw(va, ib, shader, GL_LINES);
        // render gt
        if (gt_keyframe) {
            glm::mat4 local;
            Eigen::Map<Eigen::Matrix4f> map_local(&local[0][0]);
            map_local = gt_keyframe->mat.cast<float>();
            shader.setUniformMat4f("u_MVP", proj * model_rotation_3 * local * scale);
            renderer.Draw(va, ib, shader, GL_LINES);
        }

        // draw reference frame
        for (const auto &k : keyframes) {
            glm::mat4 local;
            Eigen::Map<Eigen::Matrix4f> map_local(&local[0][0]);
            map_local = k->mat.cast<float>();
            shader.setUniformMat4f("u_MVP", proj * model_rotation_3 * local * scale);
            renderer.Draw(va, ib, shader, GL_LINES);

            auto dir = k->groundDir;
            auto xy = k->mat.col(3).head<2>();
            map_local = k->mat.cast<float>()* my_utils::get4by4FromPlane(dir, Eigen::Vector2d{0,0}).cast<float>();

            shader.setUniformMat4f("u_MVP", proj * model_rotation_3 * local * scale);
            renderer.Draw(va, ib, shader, GL_LINES);


        }

        for (const auto &p : trajectory_interpolated){
            glm::mat4 local;
            glm::mat4 scale = glm::mat4(0.005f);
            Eigen::Map<Eigen::Matrix4f> map_local(&local[0][0]);
            map_local = p.cast<float>();
            shader.setUniformMat4f("u_MVP", proj * model_rotation_3 * local * scale);
            renderer.Draw(va, ib, shader, GL_LINES);
        }
        GLCall(glPointSize(1));
        GLCall(glLineWidth(3));

        for ( int i =0; i < keyframes.size(); i++) {
            if (im_draw_only_edited && i > im_edited_frame){
                continue;
            }
            const auto &k = keyframes[i];
            shader_pc.Bind();
            glm::mat4 local;
            Eigen::Map<Eigen::Matrix4f> map_local(&local[0][0]);
            map_local = k->mat.cast<float>();

            if (i == im_edited_frame ){
                shader_pc.setUniform4f("u_COLORPC", 1,0,0,1);
            }else{

                shader_pc.setUniform4f("u_COLORPC", k->color.x(),k->color.y(),k->color.z(),1.0f);
            }
            shader_pc.setUniformMat4f("u_MVPPC", proj * model_rotation_3 * local );
            renderer.DrawArray(k->va, shader_pc, GL_POINTS, k->cloud->size());
        }

        if (gt_keyframe) {
            shader_pc.Bind();
            glm::mat4 local;
            Eigen::Map<Eigen::Matrix4f> map_local(&local[0][0]);
            map_local = gt_keyframe->mat.cast<float>();
            shader_pc.setUniform4f("u_COLORPC", 0,0,1,1);
            shader_pc.setUniformMat4f("u_MVPPC", proj * model_rotation_3 * local );
            renderer.DrawArray(gt_keyframe->va, shader_pc, GL_POINTS, gt_keyframe->cloud->size());
        }

        ImGui::Begin("SLAM Demo");
        ImGui::Checkbox("TopOrtho", &im_top_ortho);
        ImGui::SameLine();

        ImGui::InputFloat("OrthoScale", &im_ortho_scale,1.f,10.f);
        ImGui::InputFloat("OrthoSlice", &im_ortho_slice,0.1f,1.f);
        ImGui::InputFloat("OrthoHeight", &im_ortho_height,0.1f,1.f);


        ImGui::InputInt("Edited_frame",&im_edited_frame);
        ImGui::SameLine();
        if(ImGui::Button("hide gizmo")){
            im_edited_frame = -1;
        }
        if(ImGui::Button("load")) {
            std::vector<Eigen::Matrix4d> m_trajectory;
            //m_trajectory.resize(keyframes.size());
            my_utils::LoadState(json_config, m_trajectory, icp_gt_resutls);
            for (int i =0; i < m_trajectory.size(); i++)
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
            my_utils::saveState(json_config, m_trajectory, icp_gt_resutls);
        }
        ImGui::SameLine();
        if(ImGui::Button("iterpolate-slerp")) {
            trajectory_interpolated.clear();
            trajectory_ts_interpolated.clear();
            for (int i =0; i< keyframes.size()-1; i++){
                const Eigen::Affine3d curr(keyframes[i]->mat);
                const Eigen::Affine3d next(keyframes[i+1]->mat);
                const double ts1(trajectory_ts[i]);
                const double ts2(trajectory_ts[i+1]);
                const Eigen::Quaterniond q1(curr.rotation());
                const Eigen::Quaterniond q2(next.rotation());

                const Eigen::Vector3d t1(curr.translation());
                const Eigen::Vector3d t2(next.translation());


                for (float r = 0; r < 1.0;r +=0.005){
                    Eigen::Quaterniond qr = q1.slerp(r, q2);
                    Eigen::Vector3d tr = t1 + r*(t2-t1);
                    Eigen::Affine3d interpolated(Eigen::Affine3f::Identity());
                    interpolated.translate(tr);
                    interpolated.rotate(qr);
                    double ts_r = ts1 + r * (ts2 - ts1);
                    trajectory_interpolated.push_back((interpolated*laser_offset.inverse()).matrix());
                    trajectory_ts_interpolated.push_back(ts_r);
                }

                std::ofstream f("/tmp/trajectory_gt.txt");
                for (int i=0; i< trajectory_interpolated.size(); i++){
                    const Sophus::SE3d tt = Sophus::SE3d::fitToSE3(trajectory_interpolated[i]);
                    const auto tt_log = tt.log();
                    f << std::fixed << trajectory_ts_interpolated[i] << " "<< tt_log[0]<<" " << tt_log[1] << " " << tt_log[2] << " " <<tt_log[3] << " " <<tt_log[4] << " " <<tt_log[5] <<std::endl;
                }
                f.close();

            }
        }
        ImGui::SameLine();
        if(ImGui::Button("iterpolate-gtsam")) {

            std::vector<Eigen::Matrix4d> trajectory_interpolated_gtsam;
            std::vector<double> trajectory_ts_interpolated_gtsam;
            using namespace std;
            using namespace gtsam;
            NonlinearFactorGraph graph;
            for (int i =1; i < trajectory_noskip.size(); i++){
                auto odometryNoise = noiseModel::Diagonal::Variances(
                        (Vector(6) << 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1).finished());
                Eigen::Matrix4d update = trajectory_noskip[i-1].inverse() * trajectory_noskip[i];
                graph.emplace_shared<BetweenFactor<Pose3> >(i-1, i, Pose3(orthogonize(update)), odometryNoise);
            }

            for (int i =1; i < keyframes.size(); i++){
                auto priorModel = noiseModel::Diagonal::Variances(
                        (Vector(6) << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12).finished());
                graph.add(PriorFactor<Pose3>(i*6, Pose3(keyframes[i]->mat), priorModel));
            }

            graph.print("\nFactor Graph:\n");  // print
            Values initial;
            for (int i =0; i < trajectory_noskip.size(); i++){
                initial.insert(i, Pose3(orthogonize( trajectory_noskip[i])));
            }
            Values result = LevenbergMarquardtOptimizer(graph, initial).optimize();
            result.print("Final Result:\n");
            for (int i =0; i < trajectory_noskip.size(); i++){
                auto v =  result.at<Pose3>(i);
                Eigen::Matrix4d c = Eigen::Matrix4d::Identity();
                trajectory_ts_interpolated_gtsam.push_back(trajectory_ts_noskip[i]);
                trajectory_interpolated_gtsam.push_back(v.matrix());
            }

            trajectory_interpolated.clear();
            trajectory_ts_interpolated.clear();
            for (int i =0; i< trajectory_interpolated_gtsam.size()-1; i++){
                const Eigen::Affine3d curr(trajectory_interpolated_gtsam[i]);
                const Eigen::Affine3d next(trajectory_interpolated_gtsam[i+1]);
                const double ts1(trajectory_ts_interpolated_gtsam[i]);
                const double ts2(trajectory_ts_interpolated_gtsam[i+1]);
                const Eigen::Quaterniond q1(curr.rotation());
                const Eigen::Quaterniond q2(next.rotation());

                const Eigen::Vector3d t1(curr.translation());
                const Eigen::Vector3d t2(next.translation());

                for (float r = 0; r < 1.0;r +=0.05){
                    Eigen::Quaterniond qr = q1.slerp(r, q2);
                    Eigen::Vector3d tr = t1 + r*(t2-t1);
                    Eigen::Affine3d interpolated(Eigen::Affine3f::Identity());
                    interpolated.translate(tr);
                    interpolated.rotate(qr);
                    double ts_r = ts1 + r * (ts2 - ts1);
                    trajectory_interpolated.push_back((interpolated*laser_offset.inverse()).matrix());
                    trajectory_ts_interpolated.push_back(ts_r);
                }
            }

            std::ofstream f("/tmp/trajectory_gt.txt");
            for (int i=0; i< trajectory_interpolated.size(); i++){
                const Sophus::SE3d tt = Sophus::SE3d::fitToSE3(trajectory_interpolated[i]);
                const auto tt_log = tt.log();
                f << std::fixed << trajectory_ts_interpolated[i] << " "<< tt_log[0]<<" " << tt_log[1] << " " << tt_log[2] << " " <<tt_log[3] << " " <<tt_log[4] << " " <<tt_log[5] <<std::endl;
            }
            f.close();

        }
        if(ImGui::Button("flatten")) {
            using namespace std;
//            using namespace gtsam;
//
//            for (int i =0; i <keyframes.size() ; i++){
//                keyframes[i]->mat.col(3).z() = 0;
//            }


            using namespace std;
            using namespace gtsam;
            NonlinearFactorGraph graph;

            for (int i =0; i < keyframes.size(); i++){
                auto priorModel = noiseModel::Diagonal::Variances(
                        (Vector(6) << 1e12, 1e12, 1e12, 1e12, 1e12, 1e-12).finished());
                Eigen::Matrix4d mat = keyframes[i]->mat;
                mat.col(3).z() = 0;
                graph.add(PriorFactor<Pose3>(i, Pose3(mat), priorModel));
            }


            for (int i =1; i < keyframes.size(); i++){
                auto odometryNoise = noiseModel::Diagonal::Variances(
                        (Vector(6) << 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1).finished());
                Eigen::Matrix4d update = trajectory[i-1].inverse() * trajectory[i];
                graph.emplace_shared<BetweenFactor<Pose3> >(i-1, i, Pose3(orthogonize(update)), odometryNoise);
            }
            graph.print("\nFactor Graph:\n");  // print
            Values initial;
            for (int i =0; i <keyframes.size(); i++){
                initial.insert(i, Pose3(orthogonize(keyframes[i]->mat)));
            }
            Values result = LevenbergMarquardtOptimizer(graph, initial).optimize();
            result.print("Final Result:\n");
            for (int i =0; i < keyframes.size(); i++){
                auto v =  result.at<Pose3>(i);
                keyframes[i]->mat =v.matrix();
            }


        }

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
        if(ImGui::Button("reset view")){
            rot_x =0.0f;
            rot_y =0.0f;
            view_translation = glm::vec3{ 0,0,-30 };
            view_translation = glm::vec3{ 0,0,-30 };
        }

        ImGui::SliderFloat("im_ndt_res", &im_ndt_res, 0.f, 10.f);
        ImGui::SliderFloat("im_ndt_res_loop", &im_ndt_res_loop, 0.f, 10.f);
        ImGui::InputInt("im_frames_odom",&im_frames_odom);
        ImGui::Checkbox("im_run_ndt", &im_run_ndt);
        ImGui::SliderFloat("im_loop", &im_loop, 0.f, 50.f);
        if (ImGui::Button("ndt laser odometry"))
        {
            if(im_run_ndt) {
                register_ndt(keyframes, icp_results, im_ndt_res, im_loop, im_ndt_res_loop, im_frames_odom);
            }
            using namespace std;
            using namespace gtsam;
            NonlinearFactorGraph graph;
            for (auto const& x : icp_gt_resutls ){
                auto priorModel = noiseModel::Diagonal::Variances(
                        (Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished());

                graph.add(PriorFactor<Pose3>(x.first, Pose3(x.second), priorModel));
            }
            //apply gnss
//            for (int i =0; i < keyframes.size(); i++) {
//                if (keyframes[i]->gnss_valid) {
//                    std::cout << "gnss valid" << std::endl;
//                    auto priorModel = noiseModel::Diagonal::Variances(
//                            (Vector(3) <<0.07, 0.07,0.07).finished());
//                    auto meas = (Vector(3) <<keyframes[i]->UTM_offset.y(), keyframes[i]->UTM_offset.x(),0.0).finished();
//                    graph.add(GPSFactor(i,meas, priorModel));
//                }
//            }
//
            //apply imu
//            for (int i =0; i < keyframes.size(); i++) {
//                auto priorModel = noiseModel::Diagonal::Variances(
//                        (Vector(6) <<1e-5, 1e-5, 1e10, 1e10, 1e10, 1e10).finished());
//
//                graph.add(PriorFactor<Pose3>(i, Pose3(trajectory[i]), priorModel));
//            }

            // glue to ground
            for (int i =0; i < keyframes.size(); i++){
                auto priorModel = noiseModel::Diagonal::Variances(
                        (Vector(6) << 1e5, 1e5, 1e5, 1e5, 1e5, 1).finished());
                Eigen::Matrix4d mat = keyframes[i]->mat;
                mat.col(3).z() = 0;
                graph.add(PriorFactor<Pose3>(i, Pose3(mat), priorModel));
            }

            //aply odometry
            for (int i =1; i < keyframes.size(); i++){
                auto odometryNoise = noiseModel::Diagonal::Variances(
                        (Vector(6) << 0.1, 0.1, 0.5, 1,1,1).finished());
                Eigen::Matrix4d update = trajectory[i-1].inverse() * trajectory[i];
                graph.emplace_shared<BetweenFactor<Pose3> >(i-1, i, Pose3(orthogonize(update)), odometryNoise);
            }
            for (int i =0; i < icp_results.size();i++)
            {
                auto odometryNoise = noiseModel::Diagonal::Variances(
                        (Vector(6) << 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3).finished());
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
            result.print("Final Result:\n");
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
            pcl::PointCloud<pcl::PointXYZRGB> result;
            for (int i = 0; i < keyframes.size(); i++) {
                pcl::PointCloud<pcl::PointXYZRGB> partial;
                const auto &mat  = keyframes[i]->mat;
                pcl::transformPointCloud(*(keyframes[i]->cloud), partial, mat.cast<float>());
                result += partial;
            }
            pcl::io::savePCDFileBinary("/tmp/cloud.pcd", result);
        }

        if (ImGui::Button("multiview ICP - analitcal")) {
            std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> transformed;
            std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> non_transformed;
            std::vector<pcl::KdTreeFLANN<pcl::PointXYZRGB>::Ptr> transformed_kdtree;

            ceres::Problem problem;
            std::vector<Sophus::Vector6d> se3params;
            se3params.resize(keyframes.size());
            for (int i = 0; i < keyframes.size(); i++) {
                se3params[i]=Sophus::SE3d(Sophus::SE3d::fitToSE3(keyframes[i]->mat)).log();
                problem.AddParameterBlock(se3params[i].data(), Sophus::SE3d::DoF,
                                          new LocalParameterizationSE32());
            }
            for (int i = 0; i < keyframes.size(); i++) {
                const auto &mat  = keyframes[i]->mat;

                pcl::PointCloud<pcl::PointXYZRGB>::Ptr subsample(new pcl::PointCloud<pcl::PointXYZRGB>());
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_subsample(new pcl::PointCloud<pcl::PointXYZRGB>());

                pcl::ApproximateVoxelGrid<pcl::PointXYZRGB> approximate_voxel_filter;
                approximate_voxel_filter.setLeafSize (im_ndt_res/2,im_ndt_res/2,im_ndt_res/2);
                approximate_voxel_filter.setInputCloud (keyframes[i]->cloud);
                approximate_voxel_filter.filter (*subsample);
                non_transformed.push_back(subsample);

                pcl::transformPointCloud(*subsample, *transformed_subsample, mat.cast<float>());
                transformed.push_back(transformed_subsample);

                pcl::KdTreeFLANN<pcl::PointXYZRGB>::Ptr partial_subsample_kdtree(new pcl::KdTreeFLANN<pcl::PointXYZRGB>());
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
                    if (d<im_loop) {
                        pairs.push_back(std::pair<int, int>(i, j));
                    }
                }
            }
            std::mutex ceres_mtx;
            tbb::parallel_for(tbb::blocked_range<size_t>(1,pairs.size()),[&](const tbb::blocked_range<size_t>& r) {
                for (long i = r.begin(); i < r.end(); ++i) {
                    const auto  &pair  = pairs[i];
                    const auto pp = transformed[pair.first];
                    const auto pk = transformed_kdtree[pair.second];
                    for (int p1_index = 0; p1_index < pp->size(); p1_index++) {
                        pcl::PointXYZRGB pt1 = pp->at(p1_index);
                        std::vector<int> pointIdxRadiusSearch;
                        std::vector<float> pointRadiusSquaredDistance;
                        if (pk->radiusSearch(pt1, im_ndt_res, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) {
                            int p2_index = pointIdxRadiusSearch[0];
                            Eigen::Vector4f p1 = non_transformed[pair.first]->at(p1_index).getVector4fMap();
                            Eigen::Vector4f p2 = non_transformed[pair.second]->at(p2_index).getVector4fMap();
                            ceres::LossFunction *loss = new ceres::CauchyLoss(0.2);
                            //ceres::CostFunction *cost_function =costFunICP::Create(p1, p2);
                            ceres::CostFunction *cost_function = new costFunICP2(p1, p2);
                            std::lock_guard<std::mutex> lck(ceres_mtx);
                            problem.AddResidualBlock(cost_function, loss, se3params[pair.first].data(),
                                                     se3params[pair.second].data());
                        }
                    }
                }
            });
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            options.minimizer_progress_to_stdout = true;
            options.max_num_iterations = 50;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            std::cout << summary.FullReport() << "\n";

            for (int i = 0; i < keyframes.size(); i++) {\
                std::cout << "Update " << i << std::endl;
                auto t =Sophus::SE3d(Sophus::SE3d::fitToSE3(keyframes[i]->mat)).log();
                std::cout << "\t" << (t - se3params[i]).transpose() << std::endl;
                keyframes[i]->mat = Sophus::SE3d::exp(se3params[i]).matrix();
            }

        }
        if (ImGui::Button("multiview ICP")) {
            std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> transformed;
            std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> non_transformed;
            std::vector<pcl::KdTreeFLANN<pcl::PointXYZRGB>::Ptr> transformed_kdtree;

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

                pcl::PointCloud<pcl::PointXYZRGB>::Ptr subsample(new pcl::PointCloud<pcl::PointXYZRGB>());
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_subsample(new pcl::PointCloud<pcl::PointXYZRGB>());

                pcl::ApproximateVoxelGrid<pcl::PointXYZRGB> approximate_voxel_filter;
                approximate_voxel_filter.setLeafSize (0.01,0.1,0.01);
                approximate_voxel_filter.setInputCloud (keyframes[i]->cloud);
                approximate_voxel_filter.filter (*subsample);
                non_transformed.push_back(subsample);

                pcl::transformPointCloud(*subsample, *transformed_subsample, mat.cast<float>());
                transformed.push_back(transformed_subsample);

                pcl::KdTreeFLANN<pcl::PointXYZRGB>::Ptr partial_subsample_kdtree(new pcl::KdTreeFLANN<pcl::PointXYZRGB>());
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
            std::mutex ceres_mtx;
            tbb::parallel_for(tbb::blocked_range<size_t>(1,pairs.size()),[&](const tbb::blocked_range<size_t>& r) {
                for (long i = r.begin(); i < r.end(); ++i) {
                    const auto &pair = pairs[i];
                    const auto pp = transformed[pair.first];
                    const auto pk = transformed_kdtree[pair.second];
                    for (int p1_index = 0; p1_index < pp->size(); p1_index++) {
                        pcl::PointXYZRGB pt1 = pp->at(p1_index);
                        std::vector<int> pointIdxRadiusSearch;
                        std::vector<float> pointRadiusSquaredDistance;
                        if (pk->radiusSearch(pt1, 0.4, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) {
                            int p2_index = pointIdxRadiusSearch[0];
                            Eigen::Vector4f p1 = non_transformed[pair.first]->at(p1_index).getVector4fMap();
                            Eigen::Vector4f p2 = non_transformed[pair.second]->at(p2_index).getVector4fMap();
                            ceres::LossFunction *loss = new ceres::CauchyLoss(0.2);
                            ceres::CostFunction *cost_function = costFunICP::Create(p1, p2);
                            problem.AddResidualBlock(cost_function, loss, se3params[pair.first].data(),
                                                     se3params[pair.second].data());
                        }
                    }
                }
            });
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
        ImGui::Checkbox("draw edited only", &im_draw_only_edited);

        if (gt_keyframe) {
            if (ImGui::Button("register current scan to gt")) {
                Eigen::Matrix4d increment = keyframes.at(im_edited_frame)->mat;
                pcl::NormalDistributionsTransform<pcl::PointXYZRGB, pcl::PointXYZRGB> ndt;
                // Setting scale dependent NDT parameters
                // Setting minimum transformation difference for termination condition.
                ndt.setTransformationEpsilon(0.01);
                // Setting maximum step size for More-Thuente line search.
                ndt.setStepSize(0.1);
                //Setting Resolution of NDT grid structure (VoxelGridCovariance).
                ndt.setResolution(0.5);

                ndt.setMaximumIterations(100);

                ndt.setInputSource(keyframes.at(im_edited_frame)->cloud);
                ndt.setInputTarget(gt_keyframe->cloud);

                pcl::PointCloud<pcl::PointXYZRGB> t;
                Eigen::Matrix4f increment_f = increment.cast<float>();
                ndt.align(t, increment_f);

                if (ndt.hasConverged()) {
                    imgizmo = ndt.getFinalTransformation().cast<float>();
                    keyframes[im_edited_frame]->mat = ndt.getFinalTransformation().cast<double>();
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("add constraint")) {
                icp_gt_resutls[im_edited_frame] =  keyframes[im_edited_frame]->mat;
            }
            ImGui::SameLine();
            if (ImGui::Button("relax to gt")){
                using namespace std;
                using namespace gtsam;
                NonlinearFactorGraph graph;
                for (auto const& x : icp_gt_resutls ){
                    auto priorModel = noiseModel::Diagonal::Variances(
                            (Vector(6) << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12).finished());

                    graph.add(PriorFactor<Pose3>(x.first, Pose3(x.second), priorModel));
                }
                for (int i =1; i < keyframes.size(); i++){
                    auto odometryNoise = noiseModel::Diagonal::Variances(
                            (Vector(6) << 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1).finished());
                    Eigen::Matrix4d update = trajectory[i-1].inverse() * trajectory[i];
                    graph.emplace_shared<BetweenFactor<Pose3> >(i-1, i, Pose3(orthogonize(update)), odometryNoise);
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
            }
            for (auto const& x : icp_gt_resutls ){
                ImGui::Text("constraint %d", x.first);
                ImGui::SameLine();
                if(ImGui::Button(("r"+std::to_string(x.first)).c_str())){
                    const auto it=icp_gt_resutls.find(x.first);
                    icp_gt_resutls.erase(it);
                    break;
                }
            }
        }
        if(im_edited_frame > 0 && im_edited_frame < keyframes.size()){
            ImGui::Text("%s", keyframes[im_edited_frame]->fn.c_str());
        }
        ImGui::End();

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