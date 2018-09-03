#include "TComCnnFilter.h"
#include "TComTrQuant.h"




#if CNN_FILTER
TComCNNFilter::TComCNNFilter()
{
  isInitDone = 0;
  cnnfWidth = 0;
  cnnfHeight = 0;
}

TComCNNFilter::~TComCNNFilter()
{

}


void TComCNNFilter::destroy()
{

  m_picYUVBLOCKTemp.destroy();
#if CNNF_SAO_RDO
  m_picYUVcnnfTemp.destroy();
#endif
}

void TComCNNFilter::initCaffeModel(Bool useGPU,
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
  Int iQPOffsetI)
{
  //::google::InitGoogleLogging("TAppEncode");
  if (isInitDone == 0)
  {
    if (useGPU == 1)
    {
      Caffe::SetDevice(GpuId);
      Caffe::set_mode(Caffe::GPU);
    }
    else
    {
      Caffe::set_mode(Caffe::CPU);
    }
    initModel(model_fileI, trained_fileI);
    isInitDone = 1;
  }
  if (cnnfWidth != width || cnnfHeight != height)
  {
    reshapeModel(width, height);

    m_picYUVBLOCKTemp.destroy();
    m_picYUVBLOCKTemp.create(width, height, chromaFormatIDC, uiMaxCUWidth, uiMaxCUHeight, uiMaxCUDepth, true);
#if CNNF_SAO_RDO
    m_picYUVcnnfTemp.destroy();
    m_picYUVcnnfTemp.create(width, height, chromaFormatIDC, uiMaxCUWidth, uiMaxCUHeight, uiMaxCUDepth, true);
#endif

    cnnfWidth = width;
    cnnfHeight = height;
  }

  baseQp = iBaseQP;
  baseQpI = iBaseQP + iQPOffsetI;
}

void TComCNNFilter::initModel(const string& model_fileI,
                              const string& trained_fileI)
{
  // Init Y Model
  // Load the network. 
  netYUV.reset(new Net<float>(model_fileI, TEST));
  netYUV->CopyTrainedLayersFrom(trained_fileI);
}

void TComCNNFilter::BestLbpSize(const int total_width,
  const int total_height,
  int &width,
  int &height)
{
  int num = (total_width >> 1) / (2 * BLOCK_OVERLAP) + 1;
  int max_extra = total_width;
  int temp;
  int ability = min(BLOCK_ABILITY, total_width >> 1);
  width = total_width >> 1;
  height = total_height >> 1;
  for (int i = 2 * BLOCK_OVERLAP * 2; i <= ability; i++)
  {
    int extra_filter = 0;
    temp = i;
    extra_filter = (total_width - 2 * BLOCK_OVERLAP) / (temp - BLOCK_OVERLAP * 2) * BLOCK_OVERLAP * 2;
    extra_filter += temp - (total_width - 2 * BLOCK_OVERLAP) % (temp - BLOCK_OVERLAP * 2);
    extra_filter += (((total_width >> 1) - 2 * BLOCK_OVERLAP) / (temp - BLOCK_OVERLAP * 2) * BLOCK_OVERLAP * 2) * 2;
    extra_filter += (temp - ((total_width >> 1) - 2 * BLOCK_OVERLAP) % (temp - BLOCK_OVERLAP * 2)) * 2;
    if (extra_filter < max_extra)
    {
      max_extra = extra_filter;
      width = temp;
    }
  }

  num = (total_height >> 1) / (2 * BLOCK_OVERLAP) + 1;
  max_extra = total_height;
  ability = min(BLOCK_ABILITY, total_height >> 1);
  for (int i = 2 * BLOCK_OVERLAP * 2; i <= ability; i++)
  {
    int extra_filter = 0;
    temp = i;
    extra_filter = (total_height - 2 * BLOCK_OVERLAP) / (temp - BLOCK_OVERLAP * 2) * BLOCK_OVERLAP * 2;
    extra_filter += temp - (total_height - 2 * BLOCK_OVERLAP) % (temp - BLOCK_OVERLAP * 2);
    extra_filter += (((total_height >> 1) - 2 * BLOCK_OVERLAP) / (temp - BLOCK_OVERLAP * 2) * BLOCK_OVERLAP * 2) * 2;
    extra_filter += (temp - ((total_height >> 1) - 2 * BLOCK_OVERLAP) % (temp - BLOCK_OVERLAP * 2)) * 2;
    if (extra_filter < max_extra)
    {
      max_extra = extra_filter;
      height = temp;
    }
  }
}



void TComCNNFilter::reshapeModel(int width, int height)
{
  Blob<float>* input_layer;
  int min_size;

  min_size = min(height, width);
  heightBLOCK = min(min_size, BLOCK_SIZE);
  heightBLOCK = min(heightBLOCK, min_size >> 1);

  widthBLOCK = width >> 1;
  heightBLOCK = height >> 1;

  BestLbpSize(width, height, widthBLOCK, heightBLOCK);

  stride_h = heightBLOCK - 2 * BLOCK_OVERLAP;
  stride_w = widthBLOCK - 2 * BLOCK_OVERLAP;

  int test = netYUV->num_inputs();
  for (int i = 0; i < netYUV->num_inputs(); i++)
  {
    input_layer = netYUV->input_blobs()[i];
    CHECK(input_layer->channels() == 1) << "Input layer should have 1 channel.";

    input_layer->Reshape(1, 1, heightBLOCK, widthBLOCK);
  }
  netYUV->Reshape();
}



void TComCNNFilter::CNNFilterI(TComPic* pcPic)
{
  TComPicYuv *pcPicYuvRec = pcPic->getPicYuvRec();
#if !CNNF_SAO_RDO
  Pel        *piSrcY = pcPicYuvRec->getAddr(COMPONENT_Y, 0, 0);
  Pel        *piSrcCb = pcPicYuvRec->getAddr(COMPONENT_Cb, 0, 0);
  Pel        *piSrcCr = pcPicYuvRec->getAddr(COMPONENT_Cr, 0, 0);

  Pel        *piTmpSrcY = piSrcY;
  Pel        *piTmpSrcCb = piSrcCb;
  Pel        *piTmpSrcCr = piSrcCr;
#endif

  Int widthY = pcPicYuvRec->getWidth(COMPONENT_Y);
  Int heightY = pcPicYuvRec->getHeight(COMPONENT_Y);
  Int widthUV = pcPicYuvRec->getWidth(COMPONENT_Cb);
  Int heightUV = pcPicYuvRec->getHeight(COMPONENT_Cb);

  Int  iStrideY = pcPicYuvRec->getStride(COMPONENT_Y);
  Int  iStrideCb = pcPicYuvRec->getStride(COMPONENT_Cb);
  Int  iStrideCr = pcPicYuvRec->getStride(COMPONENT_Cr);

  Int lumaBitDepth = pcPic->getPicSym()->getSPS().getBitDepth(CHANNEL_TYPE_LUMA);
  Int chromaBitDepth = pcPic->getPicSym()->getSPS().getBitDepth(CHANNEL_TYPE_CHROMA);

  pcPicYuvRec->copyToPic(&m_picYUVBLOCKTemp);
#if !CNNF_SAO_RDO
  CNNFilterY(m_picYUVBLOCKTemp.getAddr(COMPONENT_Y, 0, 0), iStrideY, widthY, heightY, lumaBitDepth, COMPONENT_Y, piTmpSrcY);
  CNNFilterY(m_picYUVBLOCKTemp.getAddr(COMPONENT_Cb, 0, 0), iStrideCb, widthUV, heightUV, chromaBitDepth, COMPONENT_Cb, piTmpSrcCb);
  CNNFilterY(m_picYUVBLOCKTemp.getAddr(COMPONENT_Cr, 0, 0), iStrideCr, widthUV, heightUV, chromaBitDepth, COMPONENT_Cr, piTmpSrcCr);
#else
  CNNFilterY(m_picYUVBLOCKTemp.getAddr(COMPONENT_Y, 0, 0), iStrideY, widthY, heightY, lumaBitDepth, COMPONENT_Y,m_picYUVcnnfTemp.getAddr(COMPONENT_Y, 0, 0));
  CNNFilterY(m_picYUVBLOCKTemp.getAddr(COMPONENT_Cb, 0, 0), iStrideCb, widthUV, heightUV, chromaBitDepth, COMPONENT_Cb, m_picYUVcnnfTemp.getAddr(COMPONENT_Cb, 0, 0));
  CNNFilterY(m_picYUVBLOCKTemp.getAddr(COMPONENT_Cr, 0, 0), iStrideCr, widthUV, heightUV, chromaBitDepth, COMPONENT_Cr, m_picYUVcnnfTemp.getAddr(COMPONENT_Cr, 0, 0));
#endif
}

// get patch from Src
void TComCNNFilter::PreProcess(Pel *piSrc,
                               const int SrcStride,
                               float* pfDst,
                               const int DstStride,
                               const int width,
                               const int height,
                               ComponentID compID,
                               Int bitdepth)

{
  Pel   *pSrc = piSrc;
  Float *pDst = pfDst;
  Pel temp = 0;
  Int scale = (1 << bitdepth) - 1;
  for (int i = 0; i < height; i++)
  {
    for (int j = 0; j < width; j++)
    {
      temp = Clip3(Pel(0), Pel(scale), pSrc[j]);
      pDst[j] = (Float)((temp * 1.0) / scale);
    }
    pSrc += SrcStride;
    pDst += DstStride;
  }
}

// add patch to Dst
void TComCNNFilter::PostProcess(const float* pfSrc,
                                const int SrcStride,
                                Pel *piDst,
                                const int DstStride,
                                const int width,
                                const int height,
                                ComponentID compID,
                                Int bitdepth)

{
  Pel         temp = 0;
  Pel         *pDst = piDst;
  const Float *pSrc = pfSrc;
  Int scale = (1 << bitdepth) - 1;

  for (int i = 0; i < height; i++)
  {
    for (int j = 0; j < width; j++)
    {
      temp = (Pel)((pSrc[j] * scale) + 0.5);
      temp = Clip3(Pel(0), Pel(scale), temp);
      pDst[j] = temp;
    }
    pSrc += SrcStride;
    pDst += DstStride;
  }
}


void TComCNNFilter::GenerateQPMask(const int width,
                                   const int height,
                                   float* qpMask,
                                   Bool isIntra)
{
  int i = 0;
  int j = 0;
  float temp_qp = 0;
  float* pDest = qpMask;
  if (isIntra)
  {
    temp_qp = (float)(baseQpI * 1.0 / 51);
  }
  else
  {
    temp_qp = (float)(baseQp * 1.0 / 51);
  }

  for (int ii = 0; ii < height; ii++)
  {
    for (int jj = 0; jj < width; jj++)
    {
      pDest[jj] = temp_qp;
    }
    pDest += width;
  }
}

void TComCNNFilter::CNNFilterY( Pel *piSrc,
                                Int iStrideY,
                                Int widthY,
                                Int heightY,
                                Int lumaBitDepth,
                                ComponentID compID,
                                Pel *PiDst)
{
  CNNFilterYTop(piSrc, iStrideY, widthY, heightY, lumaBitDepth, compID, PiDst);

  Int end_height = CNNFilterYMid(piSrc, iStrideY, widthY, heightY, lumaBitDepth, compID, PiDst);

  CNNFilterYBot(piSrc, iStrideY, widthY, heightY, end_height, lumaBitDepth, compID, PiDst);
}


void TComCNNFilter::CNNFilterYTop(Pel *piSrc,
                                  Int iStrideY,
                                  Int widthY,
                                  Int heightY,
                                  Int lumaBitDepth,
                                  ComponentID compID,
                                  Pel *PiDst)
{
  float *input_data1 = NULL;
  float *input_data2 = NULL;
  const float* output_data = NULL;
  int line_num = 0;
  int line_num_x = 0;
  int end_width = 0;
  Pel *pRecPre = piSrc;
  Pel *pRecPost = PiDst;
  const float *outputDataTemp = NULL;
  Pel *pTemp = NULL;
  Pel *m_RecPreTemp = NULL;
  Pel *m_PostTemp = NULL;
  Pel *pPreTemp = NULL;
  // top 
  Pel *pRecPreTemp = pRecPre;
  Pel *pRecPostTemp = pRecPost;

  // first line    
  input_data1 = netYUV->input_blobs()[0]->mutable_cpu_data();
  input_data2 = netYUV->input_blobs()[1]->mutable_cpu_data();
  GenerateQPMask(widthBLOCK, heightBLOCK, input_data1, 1);

  PreProcess(pRecPreTemp, iStrideY, input_data2, widthBLOCK, widthBLOCK, heightBLOCK, compID, lumaBitDepth);


  netYUV->Forward();
  output_data = netYUV->output_blobs()[0]->cpu_data();

  outputDataTemp = output_data;
  pTemp = pRecPostTemp;

  PostProcess(outputDataTemp, widthBLOCK, pTemp, iStrideY, widthBLOCK, heightBLOCK, compID, lumaBitDepth);
  //first line middle
  if (widthY - (widthBLOCK) > 0)
  {
    line_num = floor((widthY - (widthBLOCK - 2 * BLOCK_OVERLAP) - 2 * BLOCK_OVERLAP) / stride_w);

    pRecPreTemp = pRecPre + stride_w;
    pRecPostTemp = pRecPost + stride_w;

    for (int i = 0; i < line_num; i++)
    {
      input_data1 = netYUV->input_blobs()[0]->mutable_cpu_data();
      input_data2 = netYUV->input_blobs()[1]->mutable_cpu_data();
      GenerateQPMask(widthBLOCK, heightBLOCK, input_data1, 1);

      PreProcess(pRecPreTemp, iStrideY, input_data2, widthBLOCK, widthBLOCK, heightBLOCK, compID, lumaBitDepth);

      netYUV->Forward();
      output_data = netYUV->output_blobs()[0]->cpu_data();

      outputDataTemp = output_data + BLOCK_OVERLAP;
      pTemp = pRecPostTemp + BLOCK_OVERLAP;

      PostProcess(outputDataTemp, widthBLOCK, pTemp, iStrideY, widthBLOCK - BLOCK_OVERLAP, heightBLOCK, compID, lumaBitDepth);

      pRecPreTemp += stride_w;
      pRecPostTemp += stride_w;
    }
    end_width = min(widthY - (widthBLOCK - 2 * BLOCK_OVERLAP) - stride_w * line_num, widthBLOCK);
    if (end_width < 2 * BLOCK_OVERLAP)
    {
      end_width = 0;
    }
  }
  else
  {
    end_width = 0;
  }

  if (end_width > 0)
  {

    end_width = widthBLOCK - BLOCK_OVERLAP;

    pRecPreTemp = pRecPre + widthY - widthBLOCK;
    pRecPostTemp = pRecPost + widthY - widthBLOCK;

    input_data1 = netYUV->input_blobs()[0]->mutable_cpu_data();
    input_data2 = netYUV->input_blobs()[1]->mutable_cpu_data();
    GenerateQPMask(widthBLOCK, heightBLOCK, input_data1, 1);

    PreProcess(pRecPreTemp, iStrideY, input_data2, widthBLOCK, widthBLOCK, heightBLOCK, compID, lumaBitDepth);

    netYUV->Forward();
    output_data = netYUV->output_blobs()[0]->cpu_data();
    outputDataTemp = output_data + widthBLOCK - end_width;
    pTemp = pRecPostTemp + widthBLOCK - end_width;

    PostProcess(outputDataTemp, widthBLOCK, pTemp, iStrideY, end_width, heightBLOCK, compID, lumaBitDepth);
  }
}


Int TComCNNFilter::CNNFilterYMid( Pel *piSrc,
                                  Int iStrideY,
                                  Int widthY,
                                  Int heightY,
                                  Int lumaBitDepth,
                                  ComponentID compID,
                                  Pel *PiDst)
{
  float *input_data1 = NULL;
  float *input_data2 = NULL;
  const float* output_data = NULL;
  int line_num = 0;
  int line_num_x = 0;
  int end_width = 0;
  int end_height = 0;
  Pel *pRecPre = piSrc;
  Pel *pRecPost = PiDst;
  const float *outputDataTemp = NULL;
  Pel *pTemp = NULL;
  Pel *m_RecPreTemp = NULL;
  Pel *m_PostTemp = NULL;
  Pel *pPreTemp = NULL;
  // top 
  Pel *pRecPreTemp = pRecPre;
  Pel *pRecPostTemp = pRecPost;


  // middle
  if (heightY - (heightBLOCK) > 0)
  {
    line_num = floor((heightY - (heightBLOCK - 2 * BLOCK_OVERLAP) - 2 * BLOCK_OVERLAP) / stride_h);
    for (int i = 0; i < line_num; i++)
    {
      //left
      m_RecPreTemp = pRecPre + iStrideY * stride_h * (i + 1);
      m_PostTemp = pRecPost + iStrideY * stride_h * (i + 1);

      pRecPreTemp = m_RecPreTemp;
      pRecPostTemp = m_PostTemp;

      input_data1 = netYUV->input_blobs()[0]->mutable_cpu_data();
      input_data2 = netYUV->input_blobs()[1]->mutable_cpu_data();
      GenerateQPMask(widthBLOCK, heightBLOCK, input_data1, 1);

      PreProcess(pRecPreTemp, iStrideY, input_data2, widthBLOCK, widthBLOCK, heightBLOCK, compID, lumaBitDepth);

      netYUV->Forward();
      output_data = netYUV->output_blobs()[0]->cpu_data();

      outputDataTemp = output_data + BLOCK_OVERLAP * widthBLOCK;
      pTemp = pRecPostTemp + BLOCK_OVERLAP * iStrideY;

      PostProcess(outputDataTemp, widthBLOCK, pTemp, iStrideY, widthBLOCK, heightBLOCK - BLOCK_OVERLAP, compID, lumaBitDepth);

      //middle

      if (widthY - (widthBLOCK) > 0)
      {
        line_num_x = floor((widthY - (widthBLOCK - 2 * BLOCK_OVERLAP) - 2 * BLOCK_OVERLAP) / stride_w);

        pRecPreTemp = m_RecPreTemp + stride_w;
        pRecPostTemp = m_PostTemp + stride_w;

        for (int j = 0; j < line_num_x; j++)
        {
          input_data1 = netYUV->input_blobs()[0]->mutable_cpu_data();
          input_data2 = netYUV->input_blobs()[1]->mutable_cpu_data();
          GenerateQPMask(widthBLOCK, heightBLOCK, input_data1, 1);

          PreProcess(pRecPreTemp, iStrideY, input_data2, widthBLOCK, widthBLOCK, heightBLOCK, compID, lumaBitDepth);


          netYUV->Forward();
          output_data = netYUV->output_blobs()[0]->cpu_data();

          outputDataTemp = output_data + widthBLOCK * BLOCK_OVERLAP + BLOCK_OVERLAP;
          pTemp = pRecPostTemp + iStrideY * BLOCK_OVERLAP + BLOCK_OVERLAP;

          PostProcess(outputDataTemp, widthBLOCK, pTemp, iStrideY, widthBLOCK - BLOCK_OVERLAP, heightBLOCK - BLOCK_OVERLAP, compID, lumaBitDepth);


          pRecPreTemp += stride_w;
          pRecPostTemp += stride_w;
        }
        end_width = min(widthY - (widthBLOCK - 2 * BLOCK_OVERLAP) - stride_w * line_num_x, widthBLOCK);
        if (end_width < 2 * BLOCK_OVERLAP)
        {
          end_width = 0;
        }
      }
      else
      {
        end_width = 0;
      }

      if (end_width > 0)
      {
        //end
        end_width = widthBLOCK - BLOCK_OVERLAP;
        pRecPreTemp = m_RecPreTemp + widthY - widthBLOCK;
        pRecPostTemp = m_PostTemp + widthY - widthBLOCK;

        input_data1 = netYUV->input_blobs()[0]->mutable_cpu_data();
        input_data2 = netYUV->input_blobs()[1]->mutable_cpu_data();
        GenerateQPMask(widthBLOCK, heightBLOCK, input_data1, 1);

        PreProcess(pRecPreTemp, iStrideY, input_data2, widthBLOCK, widthBLOCK, heightBLOCK, compID, lumaBitDepth);


        netYUV->Forward();
        output_data = netYUV->output_blobs()[0]->cpu_data();
        outputDataTemp = output_data + widthBLOCK * BLOCK_OVERLAP + widthBLOCK - end_width;
        pTemp = pRecPostTemp + iStrideY * BLOCK_OVERLAP + widthBLOCK - end_width;

        PostProcess(outputDataTemp, widthBLOCK, pTemp, iStrideY, end_width, heightBLOCK - BLOCK_OVERLAP, compID, lumaBitDepth);
      }
    }
    end_height = min(heightY - (heightBLOCK - 2 * BLOCK_OVERLAP) - stride_h * line_num, heightBLOCK);
    if (end_height < 2 * BLOCK_OVERLAP)
    {
      end_height = 0;
    }
  }
  else
  {
    end_height = 0;
  }
  return end_height;

}

void TComCNNFilter::CNNFilterYBot(Pel *piSrc,
                                  Int iStrideY,
                                  Int widthY,
                                  Int heightY,
                                  Int endHeight,
                                  Int lumaBitDepth,
                                  ComponentID compID,
                                  Pel *PiDst)
{
  float *input_data1 = NULL;
  float *input_data2 = NULL;
  const float* output_data = NULL;
  int line_num = 0;
  int end_width = 0;
  int end_height = endHeight;
  Pel *pRecPre = piSrc;
  Pel *pRecPost = PiDst;
  const float *outputDataTemp = NULL;
  Pel *pTemp = NULL;
  Pel *m_RecPreTemp = NULL;
  Pel *m_PostTemp = NULL;
  Pel *pPreTemp = NULL;
  // top 
  Pel *pRecPreTemp = pRecPre;
  Pel *pRecPostTemp = pRecPost;

  // bottom line
  if (end_height > 0)
  {
    //left
    end_height = heightBLOCK - BLOCK_OVERLAP;
    pRecPreTemp = pRecPre + iStrideY * (heightY - heightBLOCK);
    pRecPostTemp = pRecPost + iStrideY * (heightY - heightBLOCK);

    input_data1 = netYUV->input_blobs()[0]->mutable_cpu_data();
    input_data2 = netYUV->input_blobs()[1]->mutable_cpu_data();
    GenerateQPMask(widthBLOCK, heightBLOCK, input_data1, 1);

    PreProcess(pRecPreTemp, iStrideY, input_data2, widthBLOCK, widthBLOCK, heightBLOCK, compID, lumaBitDepth);

    netYUV->Forward();
    output_data = netYUV->output_blobs()[0]->cpu_data();
    outputDataTemp = output_data + widthBLOCK * (heightBLOCK - end_height);
    pTemp = pRecPostTemp + iStrideY * (heightBLOCK - end_height);

    PostProcess(outputDataTemp, widthBLOCK, pTemp, iStrideY, widthBLOCK, end_height, compID, lumaBitDepth);

    //middle
    if (widthY - (widthBLOCK) > 0)
    {
      line_num = floor((widthY - (widthBLOCK - 2 * BLOCK_OVERLAP) - 2 * BLOCK_OVERLAP) / stride_w);

      pRecPreTemp = pRecPre + iStrideY * (heightY - heightBLOCK) + stride_w;
      pRecPostTemp = pRecPost + iStrideY * (heightY - heightBLOCK) + stride_w;

      for (int i = 0; i < line_num; i++)
      {
        input_data1 = netYUV->input_blobs()[0]->mutable_cpu_data();
        input_data2 = netYUV->input_blobs()[1]->mutable_cpu_data();
        GenerateQPMask(widthBLOCK, heightBLOCK, input_data1, 1);

        PreProcess(pRecPreTemp, iStrideY, input_data2, widthBLOCK, widthBLOCK, heightBLOCK, compID, lumaBitDepth);


        netYUV->Forward();
        output_data = netYUV->output_blobs()[0]->cpu_data();

        outputDataTemp = output_data + widthBLOCK * (heightBLOCK - end_height) + BLOCK_OVERLAP;
        pTemp = pRecPostTemp + iStrideY * (heightBLOCK - end_height) + BLOCK_OVERLAP;

        PostProcess(outputDataTemp, widthBLOCK, pTemp, iStrideY, widthBLOCK - BLOCK_OVERLAP, end_height, COMPONENT_Y, lumaBitDepth);

        pRecPreTemp += stride_w;
        pRecPostTemp += stride_w;

      }
      end_width = min(widthY - (widthBLOCK - 2 * BLOCK_OVERLAP) - stride_w * line_num, widthBLOCK);
      if (end_width < 2 * BLOCK_OVERLAP)
      {
        end_width = 0;
      }
    }
    else
    {
      end_width = 0;
    }

    //right
    if (end_width > 0)
    {

      end_width = widthBLOCK - BLOCK_OVERLAP;
      pRecPreTemp = pRecPre + iStrideY * (heightY - heightBLOCK) + widthY - widthBLOCK;
      pRecPostTemp = pRecPost + iStrideY * (heightY - heightBLOCK) + widthY - widthBLOCK;

      input_data1 = netYUV->input_blobs()[0]->mutable_cpu_data();
      input_data2 = netYUV->input_blobs()[1]->mutable_cpu_data();
      GenerateQPMask(widthBLOCK, heightBLOCK, input_data1, 1);

      PreProcess(pRecPreTemp, iStrideY, input_data2, widthBLOCK, widthBLOCK, heightBLOCK, compID, lumaBitDepth);


      netYUV->Forward();
      output_data = netYUV->output_blobs()[0]->cpu_data();
      outputDataTemp = output_data + widthBLOCK * (heightBLOCK - end_height) + (widthBLOCK - end_width);
      pTemp = pRecPostTemp + iStrideY * (heightBLOCK - end_height) + (widthBLOCK - end_width);

      PostProcess(outputDataTemp, widthBLOCK, pTemp, iStrideY, end_width, end_height, compID, lumaBitDepth);
    }
  }
}



void TComCNNFilter::CNNFilterUV(Pel *piSrc,
                                Int iStrideUV,
                                Int widthUV,
                                Int heightUV,
                                Int chromaBitDepth,
                                ComponentID compID,
                                Pel *PiDst)
{
  float *input_data1 = NULL;
  float *input_data2 = NULL;
  const float* output_data = NULL;
  Pel *pRecPre = piSrc;
  Pel *pRecPost = PiDst;
  Int line_num = 0;
  const float *outputDataTemp = NULL;
  Pel *pTemp = NULL;
  Pel *m_RecPreTemp = NULL;
  Pel *m_PostTemp = NULL;
  Pel *pPreTemp = NULL;
  // top 
  Pel *pRecPreTemp = pRecPre;
  Pel *pRecPostTemp = pRecPost;
  Int end_height = 0;

  // first line    
  input_data1 = netYUV->input_blobs()[0]->mutable_cpu_data();
  input_data2 = netYUV->input_blobs()[1]->mutable_cpu_data();
  GenerateQPMask(widthBLOCK, heightBLOCK, input_data1, 1);

  PreProcess(pRecPreTemp, iStrideUV, input_data2, widthUV, widthUV, heightBLOCK, compID, chromaBitDepth);

  netYUV->Forward();
  output_data = netYUV->output_blobs()[0]->cpu_data();

  outputDataTemp = output_data;
  pTemp = pRecPostTemp;

  PostProcess(outputDataTemp, widthUV, pTemp, iStrideUV, widthUV, heightBLOCK - BLOCK_OVERLAP, compID, chromaBitDepth);

  if (heightUV - (heightBLOCK - 2 * BLOCK_OVERLAP) > 0)
  {
    // middle
    line_num = (heightUV - (heightBLOCK - 2 * BLOCK_OVERLAP)) / stride_h;

    pRecPreTemp = pRecPre + iStrideUV * stride_h;
    pRecPostTemp = pRecPost + iStrideUV * stride_h;

    for (int i = 0; i < line_num; i++)
    {
      input_data1 = netYUV->input_blobs()[0]->mutable_cpu_data();
      input_data2 = netYUV->input_blobs()[1]->mutable_cpu_data();
      GenerateQPMask(widthBLOCK, heightBLOCK, input_data1, 1);

      PreProcess(pRecPreTemp, iStrideUV, input_data2, widthUV, widthUV, heightBLOCK, compID, chromaBitDepth);

      netYUV->Forward();
      output_data = netYUV->output_blobs()[0]->cpu_data();

      outputDataTemp = output_data + BLOCK_OVERLAP * widthUV;
      pTemp = pRecPostTemp + BLOCK_OVERLAP * iStrideUV;

      PostProcess(outputDataTemp, widthUV, pTemp, iStrideUV, widthUV, stride_h, compID, chromaBitDepth);

      pRecPreTemp += iStrideUV * stride_h;
      pRecPostTemp += iStrideUV * stride_h;
    }
    end_height = min(heightUV - (heightBLOCK - 2 * BLOCK_OVERLAP) - stride_h * line_num, heightBLOCK);
  }


  // bottom line
  if (end_height > 0)
  {
    pRecPreTemp = pRecPre + iStrideUV * (heightUV - heightBLOCK);
    pRecPostTemp = pRecPost + iStrideUV * (heightUV - heightBLOCK);

    input_data1 = netYUV->input_blobs()[0]->mutable_cpu_data();
    input_data2 = netYUV->input_blobs()[1]->mutable_cpu_data();
    GenerateQPMask(widthBLOCK, heightBLOCK, input_data1, 1);
    PreProcess(pRecPreTemp, iStrideUV, input_data2, widthUV, widthUV, heightBLOCK, compID, chromaBitDepth);


    netYUV->Forward();
    output_data = netYUV->output_blobs()[0]->cpu_data();
    outputDataTemp = output_data + widthUV * (heightBLOCK - end_height);
    pTemp = pRecPostTemp + iStrideUV * (heightBLOCK - end_height);

    PostProcess(outputDataTemp, widthUV, pTemp, iStrideUV, widthUV, end_height, compID, chromaBitDepth);
  }
}
#endif