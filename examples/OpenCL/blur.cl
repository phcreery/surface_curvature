
__kernel void to_blur(__write_only image2d_t dstImg,
                      __read_only image2d_t srcImg, int aMaskSize) {

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

  int global_id = get_global_id(0);
  int offset = aMaskSize / 2;
  int sum_r = 0;
  int sum_g = 0;
  int sum_b = 0;
  int width = get_image_width(srcImg);
  int height = get_image_height(srcImg);
  int row_n = global_id / width;
  int col_n = global_id % width;
  uint4 color;
  int2 coord;
  int count = 0;

  for (int y = -offset; y <= offset; y++) {
    for (int x = -offset; x <= offset; x++) {
      coord = (int2)(col_n + x, row_n + y);
      if (coord.x < 0 || coord.x >= width || coord.y < 0 || coord.y >= height) {
        continue;
      }
      color = read_imageui(srcImg, sampler, coord);
      sum_r += color.x;
      sum_g += color.y;
      sum_b += color.z;
      count += 1;
    }
  }
  color.x = sum_r / count;
  color.y = sum_g / count;
  color.z = sum_b / count;
  coord = (int2)(col_n, row_n);
  write_imageui(dstImg, coord, color);
}

__kernel void to_blur_buffer(__global float *dst, __global float *src,
                             int aMaskSize) {

  int width = get_global_size(0);
  int height = get_global_size(1);
  int col_n = get_global_id(0);
  int row_n = get_global_id(1);
  int offset = aMaskSize / 2;
  float sum_r = 0;
  float sum_g = 0;
  float sum_b = 0;
  float4 color;
  int2 coord;
  int count = 0;

  for (int y = -offset; y <= offset; y++) {
    for (int x = -offset; x <= offset; x++) {
      coord = (int2)(col_n + x, row_n + y);
      if (coord.x < 0 || coord.x >= width || coord.y < 0 || coord.y >= height) {
        continue;
      }
      // coord.x = clamp(coord.x, 0, width - 1);
      // coord.y = clamp(coord.y, 0, height - 1);

      // color = read_imageui(src, sampler, coord);
      color.x = src[coord.y * width + coord.x];
      sum_r += color.x;
      sum_g += color.y;
      sum_b += color.z;
      count += 1;
    }
  }
  color.x = sum_r / count;
  color.y = sum_g / count;
  color.z = sum_b / count;
  coord = (int2)(col_n, row_n);
  // write_imageui(dstImg, coord, color);
  dst[coord.y * width + coord.x] = color.x;
}