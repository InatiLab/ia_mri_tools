# -*- coding: utf-8 -*-

import click
import nibabel
import numpy as np
from ia_mri_tools.signal_stats import signal_likelihood
from ia_mri_tools.features import input_normalization


def _check_image_compatibility(images):
    im_shape = images[0].shape
    im_affine = images[0].affine
    for im in images[1:]:
        assert (im.shape == im_shape), \
            "Image shape mismatch: {} and {} differ.".format(im.get_filename(), images[0].get_filename())
        assert np.all(im.affine == im_affine), \
            "Image affine mismatch: {} and {} differ.".format(im.get_filename(), images[0].get_filename())


@click.command()
@click.option('--threshold', type=click.FLOAT, default=0.7,
              help='Signal likelihood threshold.')
@click.option('--output', type=click.STRING, default='signal_mask.nii',
              help='Output filename for the coil signal mask.')
@click.argument('input_images', nargs=-1, type=click.STRING)
def estimate_signal_mask(input_images, threshold, output):
    """Estimate signal mask from one or more images."""

    click.echo('Estimating signal mask...')

    # open the images
    images = [nibabel.load(x) for x in input_images]

    # all other images must have matching orientation/dimensions/etc.
    _check_image_compatibility(images)

    # sum the input inputs
    im_shape = images[0].shape
    a = np.zeros(im_shape, dtype='float32')
    for im in images:
        a += im.get_data()

    # compute the signal likelihood and threshold
    mask = signal_likelihood(a) > threshold

    # write out the result in the same format and preserve the header
    out_image = type(images[0])(mask.astype(np.float32), affine=None, header=images[0].header)

    out_image.to_filename(output)

    click.echo('Wrote signal mask to {}.'.format(output))


# @click.command()
# @click.option('--output', type=click.STRING, default='coil_correction.nii',
#               help='Output filename for the coil correction.')
# @click.option('--width', type=click.INT, default=20, help='Smoothing kernel width in pixels.')
# @click.option('--scale', type=click.FLOAT, default=100.0, help='Scale for the signal value.')
# @click.argument('input_images', nargs=-1, type=click.STRING)
# def estimate_coil_correction(input_images, output, scale, width):
#     """Estimate receive coil intensity correction from one or more images."""

#     click.echo('Estimating receive coil intensity correction from {}'.format(input_images))
#     click.echo('  width: {}, scale: {}'.format(width, scale))

#     # open the images
#     images = [nibabel.load(x) for x in input_images]

#     # all other images must have matching orientation/dimensions/etc.
#     _check_image_compatibility(images)

#     # sum the input inputs
#     im_shape = images[0].shape
#     a = np.zeros(im_shape, dtype=np.float32)
#     for im in images:
#         a += im.get_data()

#     # compute the coil correction and scale it
#     c = coil_correction(a, width, scale)

#     # write out the result in the same format and preserve the header
#     out_image = type(images[0])(c, affine=None, header=images[0].header)
#     out_image.to_filename(output)

#     click.echo('Wrote receive coil intensity correction to {}.'.format(output))


# @click.command()
# @click.option('--correction', type=click.STRING, default='coil_correction.nii',
#               help='Filename for the coil correction image.')
# @click.option('--output', type=click.STRING, default='out.nii',
#               help='Output filename for the corrected image.')
# @click.argument('input_image', type=click.STRING)
# def apply_coil_correction(input_image, correction, output):
#     """Apply receive coil intensity correction."""

#     click.echo('Applying coil intensity correction from {} to {}.'.format(correction, input_image))

#     # open the images
#     im = nibabel.load(input_image)
#     corr = nibabel.load(correction)

#     # images must have matching orientation/dimensions/etc.
#     _check_image_compatibility([im, corr])

#     # load the image data and the coil correction data and apply
#     out = im.get_data().astype(np.float32) * corr.get_data().astype(np.float32)
#     print(type(out), out.dtype)

#     # write out the result in the same format and preserve the header
#     out_image = type(im)(out, affine=None, header=im.header)
#     out_image.to_filename(output)

#     click.echo('Wrote coil intensity corrected image to {}.'.format(output))


# @click.command()
# @click.option('--output', type=click.STRING, default='out.nii',
#               help='Output filename for the textures image.')
# @click.argument('input_image', type=click.STRING)
# @click.argument('nscales', type=click.INT, default=4)
# @click.argument('norm_scale', type=click.INT, default=4)
# def estimate_textures(input_image, nscales, norm_scale, output):
#     """Estimate 3D multiscale textures."""

#     click.echo('Estimate 3D multiscale textures for {}.'.format(input_image))

#     # open the images
#     im = nibabel.load(input_image)

#     # compute the textures
#     t, names = textures(im.get_data(), nscales=nscales, norm_scale=norm_scale)
#     nfeats = len(t)
#     out = np.zeros([*t[0].shape, nfeats], t[0].dtype)
#     for f in range(nfeats):
#         out[...,f] = t[f]

#     # write out the result in the same format and preserve the header
#     out_image = type(im)(out, affine=None, header=im.header)
#     out_image.to_filename(output)

#     click.echo('Wrote 3D multiscale textures to {}.'.format(output))


# @click.command()
# @click.option('--output', type=click.STRING, default='out.nii',
#               help='Output filename for the normalized image.')
# @click.argument('input_image', type=click.STRING)
# @click.option('--width', type=click.INT, default=20, help='Smoothing kernel width in pixels.')
# def normalize_local(input_image, width, output):
#     """Local contrast normalization"""

#     click.echo('Computing local contrast normalization for {}.'.format(input_image))

#     # open the images
#     im = nibabel.load(input_image)

#     # normalize
#     out = normalize_local_3d(im.get_data(), width)

#     # write out the result and preserve the header
#     out_image = type(im)(out, affine=None, header=im.header)
#     out_image.to_filename(output)

#     click.echo('Wrote 3D multiscale textures to {}.'.format(output))
