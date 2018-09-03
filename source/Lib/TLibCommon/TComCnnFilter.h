#ifndef __TCOMCNNFILTERI__
#define __TCOMCNNFILTERI__

#include "TComPic.h"

#if CNN_FILTER
#if CNNF_CPU_ONLY
#define  CPU_ONLY
#endif
#include "caffe/caffe.hpp"

#ifndef LAYER_H_
#define LAYER_H_

#ifdef _WIN32
#include "caffe/layers/input_layer.hpp"
#include "caffe/layers/conv_ristretto_layer.hpp"
#include "caffe/layers/concat_ristretto_layer.hpp"
#include "caffe/layers/eltwise_ristretto_layer.hpp"

namespace caffe
{
  extern INSTANTIATE_CLASS(InputLayer);
  REGISTER_LAYER_CLASS(ConvolutionRistretto);
  REGISTER_LAYER_CLASS(ConcatRistretto);
  REGISTER_LAYER_CLASS(EltwiseRistretto);


}
#endif

#endif

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;


class TComCNNFilter
{
public:
  TComCNNFilter();
  ~TComCNNFilter();

  void destroy();

  void CNNFilterI(TComPic* pcPic);

  void initCaffeModel(Bool useGPU,
                      const int GpuId,
                      const string& model_fileI,
                      const string& trained_fileI,
                      int width,
                      int height,
                      const ChromaFormat chromaFormatIDC, ///< chroma format
                      const UInt uiMaxCUWidth,            ///< used for generating offsets to CUs. Can use iPicWidth if no offsets are required
                      const UInt uiMaxCUHeight,           ///< used for generating offsets to CUs. Can use iPicHeight if no offsets are required
                      const UInt uiMaxCUDepth,
                      Bool useCNNFilterI,
                      Int  iBaseQP,
                      Int iQPOffsetI);


#if CNNF_SAO_RDO
  TComPicYuv* getYUVcnnfTemp()  {return &m_picYUVcnnfTemp;}
#endif
private:

  void reshapeModel(int width, int height);
  void initModel(const string& model_fileI,
                 const string& trained_fileI);
  void BestLbpSize( const int total_width,
                    const int total_height,
                    int& width,
                    int& height);

  void PreProcess(Pel *piSrc,
                  const int SrcStride,
                  float* pfDst,
                  const int DstStride,
                  const int width,
                  const int height,
                  ComponentID compID,
                  Int bitdepth);

  void PostProcess( const float* pfSrc,
                    const int SrcStride,
                    Pel *piDst,
                    const int DstStride,
                    const int width,
                    const int height,
                    ComponentID compID,
                    Int bitdepth);



  void GenerateQPMask(const int width,
                      const int height,
                      float* qpMask,
                      Bool isIntra);

  void CNNFilterY(Pel *piSrc,
                  Int iStrideY,
                  Int widthY,
                  Int heightY,
                  Int lumaBitDepth,
                  ComponentID compID,
                  Pel *PiDst);

  void CNNFilterYTop( Pel *piSrc,
                      Int iStrideY,
                      Int widthY,
                      Int HeightY,
                      Int lumaBitDepth,
                      ComponentID compID,
                      Pel *PiDst);

  Int CNNFilterYMid(Pel *piSrc,
                    Int iStrideY,
                    Int widthY,
                    Int HeightY,
                    Int lumaBitDepth,
                    ComponentID compID,
                    Pel *PiDst);


  void CNNFilterYBot( Pel *piSrc,
                      Int iStrideY,
                      Int widthY,
                      Int HeightY,
                      Int endHeight,
                      Int lumaBitDepth,
                      ComponentID compID,
                      Pel *PiDst);

  void CNNFilterUV( Pel *piSrc,
                    Int iStrideUV,
                    Int widthUV,
                    Int heightUV,
                    Int chromaBitDepth,
                    ComponentID compID,
                    Pel *PiDst);

  void GenerateRefMask( TComPic* pcPic,
                        char* refMask,
                        const int width,
                        const int height);

  void GenerateResidual(TComPic* pcPic,
                        const int width,
                        const int height,
                        float* residual,
                        ComponentID textType);

  void writePredYUV(TComPicYuv *pcPicYuvPred);
  void writeRecYUV(TComPicYuv *pcPicYuvRec);


private:


  TComPicYuv              m_picYUVBLOCKTemp;
#if CNNF_SAO_RDO
  TComPicYuv              m_picYUVcnnfTemp;
#endif

  boost::shared_ptr<Net<float> > netYUV;

  int  baseQp;
  int  baseQpI;
  int curPoc;
  Int heightBLOCK;
  Int widthBLOCK;
  Int stride_h;
  Int stride_w;
  Int isInitDone;
  Int cnnfWidth;
  Int cnnfHeight;
};

#endif

#endif



