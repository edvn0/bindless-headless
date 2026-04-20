# Finds ImageMagick 7 (Q16HDRI) and defines imported targets:
#   ImageMagick::MagickCore
#   ImageMagick::MagickWand
#   ImageMagick::Magick++
#
# Sets:
#   ImageMagick7_FOUND

include(FindPackageHandleStandardArgs)

find_path(ImageMagick7_INCLUDE_DIR
  NAMES Magick++.h
  PATH_SUFFIXES ImageMagick-7 ImageMagick)

find_library(ImageMagick7_MagickCore_LIBRARY NAMES MagickCore-7.Q16HDRI)
find_library(ImageMagick7_MagickWand_LIBRARY NAMES MagickWand-7.Q16HDRI)
find_library(ImageMagick7_MagickPP_LIBRARY   NAMES Magick++-7.Q16HDRI)

mark_as_advanced(
  ImageMagick7_INCLUDE_DIR
  ImageMagick7_MagickCore_LIBRARY
  ImageMagick7_MagickWand_LIBRARY
  ImageMagick7_MagickPP_LIBRARY)

find_package_handle_standard_args(ImageMagick7
  REQUIRED_VARS
    ImageMagick7_INCLUDE_DIR
    ImageMagick7_MagickCore_LIBRARY
    ImageMagick7_MagickWand_LIBRARY
    ImageMagick7_MagickPP_LIBRARY)

if(ImageMagick7_FOUND)
  if(NOT TARGET ImageMagick::MagickCore)
    add_library(ImageMagick::MagickCore IMPORTED SHARED)
    set_target_properties(ImageMagick::MagickCore PROPERTIES
      IMPORTED_LOCATION             "${ImageMagick7_MagickCore_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${ImageMagick7_INCLUDE_DIR}")
  endif()

  if(NOT TARGET ImageMagick::MagickWand)
    add_library(ImageMagick::MagickWand IMPORTED SHARED)
    set_target_properties(ImageMagick::MagickWand PROPERTIES
      IMPORTED_LOCATION             "${ImageMagick7_MagickWand_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${ImageMagick7_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES      "ImageMagick::MagickCore")
  endif()

  if(NOT TARGET ImageMagick::Magick++)
    add_library(ImageMagick::Magick++ IMPORTED SHARED)
    set_target_properties(ImageMagick::Magick++ PROPERTIES
      IMPORTED_LOCATION             "${ImageMagick7_MagickPP_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${ImageMagick7_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES      "ImageMagick::MagickWand;ImageMagick::MagickCore")
  endif()
endif()