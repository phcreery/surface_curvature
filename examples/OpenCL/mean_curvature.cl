

__kernel void mean_curvature(__global float *dst, __global float *dx,
                             __global float *dy, __global float *dxx,
                             __global float *dxy, __global float *dyy) {

  const int i = get_global_id(0), j = get_global_id(1);
  const int2 coord = (int2){i, j};
  // int width = get_image_width(src);
  // int height = get_image_height(src);
  int width = get_global_size(0);
  // int height = get_global_size(1);

  float dxi = dx[j * width + i];
  float dyi = dy[j * width + i];
  float dxxi = dxx[j * width + i];
  float dxyi = dxy[j * width + i];
  float dyyi = dyy[j * width + i];

  float H;
  H = (dxi * dxi + 1) * dyyi - 2 * dxi * dyi * dxyi + (dyi * dyi + 1) * dxxi;
  H = H / (2 * pow(dxi * dxi + dyi * dyi + 1, (float)1.5));

  dst[j * width + i] = H;
}