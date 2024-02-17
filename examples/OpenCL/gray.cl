
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

  // float4 color = convert_float4(pixel) / 255;
  // color.xyz = 0.2126*color.x + 0.7152*color.y + 0.0722*color.z;
  // pixel = convert_uint4_rte(color * 255);
  // write_imageui(output, coord, pixel);

  uint gray = 0.2126 * color.x + 0.7152 * color.y + 0.0722 * color.z;
  // uint gray = (color.x + color.y + color.z) / 3;
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