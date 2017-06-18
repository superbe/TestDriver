// http://docs.nvidia.com/cuda/cuda-runtime-api/index.html#ixzz4kI0W7OiI

#include "cuda_runtime.h"
#include <iostream>

int main(int argc, char *argv[])
{
	setlocale(LC_ALL, "Russian");
	cudaDeviceProp prop;
	int count;
	cudaGetDeviceCount(&count);
	printf("���������� ������������� ���������:                                                 %d\n", count);

	for (int i = 0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		printf("--------------------------------------------------------------------------------", prop.name);
		printf("\n");
		printf("�������� �� ���������� %d\n", i);
		printf("\n");
		printf("������������:                                                                       %s\n", prop.name);
		printf("��������� (Compute capability):                                                     %d.%d\n", prop.major, prop.minor);
		printf("�������� ������� � ����������:                                                      %d\n", prop.clockRate);
		printf("���������� ����������� ������������ DMA:                                            %d\n", prop.asyncEngineCount);
		printf("����-��� ���������� ����:                                                           ");
		if (prop.kernelExecTimeoutEnabled) printf("�������\n"); else printf("��������\n");
		printf("���������� GPU �������������:                                                       ");
		if (prop.integrated) printf("��\n"); else printf("���\n");
		printf("����� ����������:                                                                   %d ", prop.computeMode);
		switch (prop.computeMode)
		{
		case 1:
			printf("(�������������� ����� ����������, ������ ���� ���� � ����� �������� ������ ������������ cudaSetDevice())\n", prop.computeMode);
			break;
		case 2:
			printf("(����������� ����� ����������, �� ���� ���� �� ����� ������������ cudaSetDevice())\n", prop.computeMode);
			break;
		case 3:
			printf("(���������� ����� ����������, ��������� ����� � ����� �������� ������ ������������ cudaSetDevice())\n", prop.computeMode);
			break;
		default:
			printf("(����� ���������� �� ���������, ��������� ������� ����� ������������ cudaSetDevice())\n", prop.computeMode);
		}
		printf("���������� ������������ ���������� ���������� ���� ������������ � ����� ���������:  ");
		if (prop.concurrentKernels) printf("��\n"); else printf("���\n");


		printf("\n");

		printf("�������� � ������ ���������� %d\n", i);
		printf("\n");
		printf("������ ���������� ������ � ������:                                                  %ld\n", prop.totalGlobalMem);
		printf("������ ����������� ������ � ������:                                                 %ld\n", prop.totalConstMem);
		printf("������������ ��� ����������� � ������:                                              %ld\n", prop.memPitch);
		printf("������������ �������:                                                               %ld\n", prop.textureAlignment);
		printf("�������� ���������� ������ CPU �� �������� ������������ Cuda-����������:            ");
		if (prop.canMapHostMemory) printf("��\n"); else printf("���\n");
		printf("\n");

		printf("�������� � ����������������� ���������� %d\n", i);
		printf("\n");
		printf("���������� �����������������:                                                       %d\n", prop.multiProcessorCount);
		printf("����������� ������ �� ���� ���������������:                                         %ld\n", prop.sharedMemPerBlock);
		printf("��������� �� ���� ���������������:                                                  %d\n", prop.regsPerBlock);
		printf("����� �� ����:                                                                      %d\n", prop.warpSize);
		printf("������������ ���������� ����� � �����:                                              %d\n", prop.maxThreadsPerBlock);
		printf("������������ ���������� ����� �� ����������:                                        (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("������������ ������� �����:                                                         (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("\n");








		//printf("�������� ��������� ECC:  %d\n", prop.ECCEnabled);

			//printf(":  %d\n", prop.concurrentManagedAccess)
			//���������� ����� ���������� ������ � ����������� ������ ������������ � �����������

			//printf(":  %d\n", prop.globalL1CacheSupported)
			//���������� ������������ ����������� ������ � L1

			//printf(":  %d\n", prop.hostNativeAtomicSupported)
			//����� ����� ����������� � ������ ������������ ������ ��������� ��������

			//printf(":  %d\n", prop.isMultiGpuBoard)
			//���������� �� ���������� GPU �����

			//printf(":  %d\n", prop.l2CacheSize)
			//������ ���� L2 � ������

			//printf(":  %d\n", prop.localL1CacheSupported)
			//���������� ������������ ����������� ������� ������� � L1

			//printf(":  %d\n", prop.managedMemory)
			//������������ ���������� ������ �������� �� ����������� �� ���� �������

			//printf(":  %d\n", prop.maxSurface1D)
			//������������ ������ 1D �����������

			//��� cudaDeviceProp::maxSurface1DLayered[2])
			//������������ ������� 1D �������� �����������

			//��� cudaDeviceProp::maxSurface2D[2])
			//������������ ������� ����������� 2D

			//��� cudaDeviceProp::maxSurface2DLayered[3])
			//������������ ������� �������� 2D �����������

			//��� cudaDeviceProp::maxSurface3D[3])
			//������������ ������� ����������� 3D

			//printf(":  %d\n", prop.maxSurfaceCubemap)
			//������������ ������� ����������� Cubemap

			//��� cudaDeviceProp::maxSurfaceCubemapLayered[2])
			//������������ ������� �������� Cubemap �����������

			//printf(":  %d\n", prop.maxTexture1D)
			//������������ ������ �������� 1D

			//��� cudaDeviceProp::maxTexture1DLayered[2])
			//������������ ������� 1D �������� ��������

			//printf(":  %d\n", prop.maxTexture1DLinear)
			//������������ ������ 1D ������� ������� � �������� �������

			//printf(":  %d\n", prop.maxTexture1DMipmap)
			//������������ 1D ��� ������ ��������

			//��� cudaDeviceProp::maxTexture2D[2])
			//������������ ������� 2D ��������

			//��� cudaDeviceProp::maxTexture2DGather[2])
			//������������ ������� 2D ��������, ���� �������� ��������� �������� ������ ���� ���������

			//��� cudaDeviceProp::maxTexture2DLayered[3])
			//������������ ������� 2D �������� ��������

			//��� cudaDeviceProp::maxTexture2DLinear[3])
			//������������ �������(������, ������, ������) ��� 2D �������, ��������� � ������� �������

			//��� cudaDeviceProp::maxTexture2DMipmap[2])
			//������������ ������� 2D ��� ��������

			//��� cudaDeviceProp::maxTexture3D[3])
			//������������ ������� �������� 3D

			//��� cudaDeviceProp::maxTexture3DAlt[3])
			//������������ ������� ���������� 3D ��������

			//printf(":  %d\n", prop.maxTextureCubemap)
			//������������ ������� �������� Cubemap

			//��� cudaDeviceProp::maxTextureCubemapLayered[2])
			//������������ ������� Cubemap �������� ��������

			//printf(":  %d\n", prop.maxThreadsPerMultiProcessor)
			//������������ ������ ���������� �� �����������������

			//printf(":  %d\n", prop.memoryBusWidth)
			//������ ���� ���������� ������ � �����

			//printf(":  %d\n", prop.multiGpuBoardGroupID)
			//���������� ������������� ��� ������ ��������� �� ����� � ��� �� ������ - GPU �����

			//printf(":  %d\n", prop.pageableMemoryAccess)
			//���������� ������������ ���������� ������ � ����������� ������ ��� ������ cudaHostRegister �� ���

			//printf(":  %d\n", prop.pciBusID)
			//���� PCI ������������� ����������

			//printf(":  %d\n", prop.pciDeviceID)
			//PCI - ���������� ������������� ����������

			//printf(":  %d\n", prop.pciDomainID)
			//PCI - ������������� ������ ����������

			//printf(":  %d\n", prop.regsPerMultiprocessor)
			//32 - ������ ���������, ��������� � �����������������

			//size_t cudaDeviceProp::sharedMemPerMultiprocessor)
			//����� ������, ��������� �� ����������������� � ������

			//printf(":  %d\n", prop.singleToDoublePrecisionPerfRatio)
			//����������� ������������������ ����� ��������(� ��������� � ��������� ������� � �������) �� ������� �������� ����������

			//printf(":  %d\n", prop.streamPrioritiesSupported)
			//���������� ������������ ���������� �������

			//size_t cudaDeviceProp::surfaceAlignment)
			//������������ ���������� � �����������

			//printf(":  %d\n", prop.tccDriver)
			//1, ���� ���������� �������� ����������� �����, ��������� ������� TCC, 0 � ��������� ������

			//size_t cudaDeviceProp::texturePitchAlignment)
			//��� ���������� ������������ ��� ������ ������� ������� � ������� �������

			//printf(":  %d\n", prop.unifiedAddressing)
			//���������� ��������� ������ �������� ������������ � ��������


		//printf(":  %d\n", prop.concurrentManagedAccess);
		//printf(":  %d\n", prop.ECCEnabled);
		//printf(":  %d\n", prop.globalL1CacheSupported);
		//printf(":  %d\n", prop.hostNativeAtomicSupported);
		//printf(":  %d\n", prop.isMultiGpuBoard);
		//printf(":  %d\n", prop.l2CacheSize);
		//printf(":  %d\n", prop.localL1CacheSupported);
		//printf(":  %d\n", prop.managedMemory);

		//printf(":  %d\n", prop.maxSurface1D);
		//printf(":  %d\n", prop.maxSurface1DLayered);
		//printf(":  %d\n", prop.maxSurface2D);
		//printf(":  %d\n", prop.maxSurface2DLayered);
		//printf(":  %d\n", prop.maxSurface3D);
		//printf(":  %d\n", prop.maxSurfaceCubemap);
		//printf(":  %d\n", prop.maxSurfaceCubemapLayered);

		//printf(":  %d\n", prop.maxTexture1D);
		//printf(":  %d\n", prop.maxTexture1DLayered);
		//printf(":  %d\n", prop.maxTexture1DLinear);
		//printf(":  %d\n", prop.maxTexture1DMipmap);
		//printf(":  %d\n", prop.maxTexture2D);
		//printf(":  %d\n", prop.maxTexture2DGather);
		//printf(":  %d\n", prop.maxTexture2DLayered);
		//printf(":  %d\n", prop.maxTexture2DLinear);
		//printf(":  %d\n", prop.maxTexture2DMipmap);
		//printf(":  %d\n", prop.maxTexture3D);
		//printf(":  %d\n", prop.maxTexture3DAlt);
		//printf(":  %d\n", prop.maxTextureCubemap);
		//printf(":  %d\n", prop.maxTextureCubemapLayered);

		//printf(":  %d\n", prop.maxThreadsPerMultiProcessor);
		//printf(":  %d\n", prop.memoryBusWidth);
		//printf(":  %d\n", prop.multiGpuBoardGroupID);
		//printf(":  %d\n", prop.pageableMemoryAccess);
		//printf(":  %d\n", prop.pciBusID);
		//printf(":  %d\n", prop.pciDeviceID);
		//printf(":  %d\n", prop.pciDomainID);
		//printf(":  %d\n", prop.regsPerMultiprocessor);
		//printf(":  %d\n", prop.sharedMemPerMultiprocessor);
		//printf(":  %d\n", prop.singleToDoublePrecisionPerfRatio);
		//printf(":  %d\n", prop.streamPrioritiesSupported);
		//printf(":  %d\n", prop.surfaceAlignment);
		//printf(":  %d\n", prop.tccDriver);
		//printf(":  %d\n", prop.texturePitchAlignment);
		//printf(":  %d\n", prop.unifiedAddressing);
	}

	system("pause");

	return 0;
}