void iradonT_cpu(int width, int n_angles, float center, float* angles, float* orig, float* dest);
void iradonT_gpu(int width, int n_angles, float center, float* d_angles, float* d_orig, float* d_dest);

/*
void diagRadonT_cpu(int width, int n_angles, float center, float* angles, float* dest);
float* diagRadonT_gpu(int width, int n_angles, float center, float* d_angles);
*/

