
__kernel void to_gray(__write_only image2d_t output,
                      __read_only image2d_t input) {

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

  const int i = get_global_id(0), j = get_global_id(1);
  const int2 coord = (int2){i, j};
  int2 size = get_image_dim(input);
  int width = get_image_width(input);
  int height = get_image_height(input);

  uint4 color = read_imageui(input, sampler, coord);
  uint gray = 0.2126 * color.x + 0.7152 * color.y + 0.0722 * color.z;

  write_imageui(output, coord, (uint4)(gray, gray, gray, 0));
}

__kernel void to_gray_buffer(__global float *output,
                             __read_only image2d_t input) {

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

  const int i = get_global_id(0), j = get_global_id(1);
  const int2 coord = (int2){i, j};
  int2 size = get_image_dim(input);
  int width = get_image_width(input);
  int height = get_image_height(input);

  uint4 color = read_imageui(input, sampler, coord);
  float gray = 0.2126 * color.x + 0.7152 * color.y + 0.0722 * color.z;
  // output[i * width + j] = gray;
  output[i + j * width] = gray;
}