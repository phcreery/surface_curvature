

__kernel void gradient_x_2d_img(__write_only image2d_t dst,
                                __read_only image2d_t src) {
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

  const int i = get_global_id(0), j = get_global_id(1);
  const int2 coord = (int2){i, j};
  const int2 coordA = (int2){i - 1, j};
  const int2 coordB = (int2){i + 1, j};

  int valueA = read_imageui(src, sampler, coordA).x;
  int valueB = read_imageui(src, sampler, coordB).x;
  // float4 color = convert_float4(pixel) / 255;
  int res = valueB - valueA;

  uint4 color;
  // if (res > 0) color = (uint4){res,0,0,255};
  // else if (res < 0) color = (uint4){0,0,-res,255};
  // else color = (uint4){0,0,0,255};

  color = (uint4){res / 2 + 128, res / 2 + 128, res / 2 + 128, 255};

  write_imageui(dst, coord, color);
}

__kernel void gradient_x_2d(__global float *dst, __read_only image2d_t src) {
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

  const int i = get_global_id(0), j = get_global_id(1);
  int width = get_image_width(src);
  int height = get_image_height(src);
  const int2 coord = (int2){i, j};
  const int2 coordA = (int2){i - 1, j};
  const int2 coordB = (int2){i + 1, j};

  int valueA = read_imageui(src, sampler, coordA).x;
  int valueB = read_imageui(src, sampler, coordB).x;
  float res = valueB - valueA;

  printf("val");

  // write_imagef(dst, coord, res);
  dst[i + j * width] = res;
  // dst[i + j*get_global_size(0)] = res;

  // random number between 0-1  --
  // https://stackoverflow.com/questions/9912143/how-to-get-a-random-number-in-opencl
  // float ptr = 0.0f;
  // float rand = fract(sin(i*112.9898f + j*179.233f + 1*237.212f) *
  // 43758.5453f, &ptr); dst[i + j*width] = rand;
}

__kernel void gradient_x_2d_buffer(__global float *dst, __global float *src) {
  const int i = get_global_id(0), j = get_global_id(1);
  const int width = get_global_size(0);
  const int2 coord = (int2){i, j};
  const int2 coordA = (int2){i - 1, j};
  const int2 coordB = (int2){i + 1, j};

  float valueA = src[coordA.x + coordA.y * width];
  float valueB = src[coordB.x + coordB.y * width];

  // clamp to edge if out of bounds by setting value to the same as the edge
  if (coordA.x < 0)
    valueA = src[coord.x + coord.y * width];
  if (coordB.x >= width)
    valueB = src[coord.x + coord.y * width];

  float res = valueB - valueA;

  dst[i + j * width] = res;
}

__kernel void gradient_y_2d_img(__write_only image2d_t dst,
                                __read_only image2d_t src) {
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

  const int i = get_global_id(0), j = get_global_id(1);
  const int2 coord = (int2){i, j};
  const int2 coordA = (int2){i, j - 1};
  const int2 coordB = (int2){i, j + 1};

  int valueA = read_imageui(src, sampler, coordA).x;
  int valueB = read_imageui(src, sampler, coordB).x;
  int res = valueB - valueA;

  uint4 color;
  // if (res > 0) color = (uint4){res,0,0,255};
  // else if (res < 0) color = (uint4){0,0,-res,255};
  // else color = (uint4){0,0,0,255};

  color = (uint4){res / 2 + 128, res / 2 + 128, res / 2 + 128, 255};

  write_imageui(dst, coord, color);
}

__kernel void gradient_y_2d_buffer(__global float *dst, __global float *src) {
  const int i = get_global_id(0), j = get_global_id(1);
  const int width = get_global_size(0);
  const int2 coord = (int2){i, j};
  const int2 coordA = (int2){i, j - 1};
  const int2 coordB = (int2){i, j + 1};

  float valueA = src[coordA.x + coordA.y * width];
  float valueB = src[coordB.x + coordB.y * width];

  // clamp to edge if out of bounds by setting value to the same as the edge
  if (coordA.y < 0)
    valueA = src[coord.x + coord.y * width];
  if (coordB.y >= get_global_size(1))
    valueB = src[coord.x + coord.y * width];

  float res = valueB - valueA;

  dst[i + j * width] = res;
}