// http://docs.nvidia.com/cuda/cuda-runtime-api/index.html#ixzz4kI0W7OiI

#include "cuda_runtime.h"
#include <iostream>

int main(int argc, char *argv[])
{
	setlocale(LC_ALL, "Russian");
	cudaDeviceProp prop;
	int count;
	cudaGetDeviceCount(&count);
	printf("Количество установленных устройств:                                                 %d\n", count);

	for (int i = 0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		printf("--------------------------------------------------------------------------------", prop.name);
		printf("\n");
		printf("СВЕДЕНИЯ ОБ УСТРОЙСТВЕ %d\n", i);
		printf("\n");
		printf("Наименование:                                                                       %s\n", prop.name);
		printf("Поколение (Compute capability):                                                     %d.%d\n", prop.major, prop.minor);
		printf("Тактовая частота в килогерцах:                                                      %d\n", prop.clockRate);
		printf("Количество асинхронных контроллеров DMA:                                            %d\n", prop.asyncEngineCount);
		printf("Тайм-аут выполнения ядра:                                                           ");
		if (prop.kernelExecTimeoutEnabled) printf("включен\n"); else printf("выключен\n");
		printf("Устройство GPU интегрировано:                                                       ");
		if (prop.integrated) printf("да\n"); else printf("нет\n");
		printf("Режим вычислений:                                                                   %d ", prop.computeMode);
		switch (prop.computeMode)
		{
		case 1:
			printf("(исключительный режим вычисления, только одна нить в одном процессе сможет использовать cudaSetDevice())\n", prop.computeMode);
			break;
		case 2:
			printf("(запрещенный режим вычисления, ни одна нить не может использовать cudaSetDevice())\n", prop.computeMode);
			break;
		case 3:
			printf("(совместный режим вычисления, несколько нитей в одном процессе смогут использовать cudaSetDevice())\n", prop.computeMode);
			break;
		default:
			printf("(режим вычисления по умолчанию, несколько потоков могут использовать cudaSetDevice())\n", prop.computeMode);
		}
		printf("Устройство поддерживает исполнение нескольких ядер одновременно в одном контексте:  ");
		if (prop.concurrentKernels) printf("да\n"); else printf("нет\n");


		printf("\n");

		printf("СВЕДЕНИЯ О ПАМЯТИ УСТРОЙСТВА %d\n", i);
		printf("\n");
		printf("Размер глобальной памяти в байтах:                                                  %ld\n", prop.totalGlobalMem);
		printf("Размер константной памяти в байтах:                                                 %ld\n", prop.totalConstMem);
		printf("Максимальный шаг копирования в памяти:                                              %ld\n", prop.memPitch);
		printf("Выравнивание текстур:                                                               %ld\n", prop.textureAlignment);
		printf("Возможно отображать память CPU на адресное пространство Cuda-устройства:            ");
		if (prop.canMapHostMemory) printf("да\n"); else printf("нет\n");
		printf("\n");

		printf("СВЕДЕНИЯ О МУЛЬТИПРОЦЕССОРАХ УСТРОЙСТВА %d\n", i);
		printf("\n");
		printf("Количество мультипроцессоров:                                                       %d\n", prop.multiProcessorCount);
		printf("Разделяемая память на один мультипроцессор:                                         %ld\n", prop.sharedMemPerBlock);
		printf("Регистров на один мультипроцессор:                                                  %d\n", prop.regsPerBlock);
		printf("Нитей на варп:                                                                      %d\n", prop.warpSize);
		printf("Максимальное количество нитей в блоке:                                              %d\n", prop.maxThreadsPerBlock);
		printf("Максимальное количество нитей по измерениям:                                        (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Максимальные размеры сетки:                                                         (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("\n");








		//printf("Включена поддержка ECC:  %d\n", prop.ECCEnabled);

			//printf(":  %d\n", prop.concurrentManagedAccess)
			//Устройство может когерентно доступ к управляемой памяти одновременно с процессором

			//printf(":  %d\n", prop.globalL1CacheSupported)
			//Устройство поддерживает кэширование глобал в L1

			//printf(":  %d\n", prop.hostNativeAtomicSupported)
			//Связь между устройством и хостом поддерживает родные атомарные операции

			//printf(":  %d\n", prop.isMultiGpuBoard)
			//Устройство на нескольких GPU борту

			//printf(":  %d\n", prop.l2CacheSize)
			//Размер кэша L2 в байтах

			//printf(":  %d\n", prop.localL1CacheSupported)
			//Устройство поддерживает кэширование местных жителей в L1

			//printf(":  %d\n", prop.managedMemory)
			//поддерживают устройства памяти выделяют на управляемый по этой системе

			//printf(":  %d\n", prop.maxSurface1D)
			//Максимальный размер 1D поверхность

			//ИНТ cudaDeviceProp::maxSurface1DLayered[2])
			//Максимальные размеры 1D слоистые поверхности

			//ИНТ cudaDeviceProp::maxSurface2D[2])
			//Максимальные размеры поверхности 2D

			//ИНТ cudaDeviceProp::maxSurface2DLayered[3])
			//Максимальные размеры слоистых 2D поверхности

			//ИНТ cudaDeviceProp::maxSurface3D[3])
			//Максимальные размеры поверхности 3D

			//printf(":  %d\n", prop.maxSurfaceCubemap)
			//Максимальные размеры поверхности Cubemap

			//ИНТ cudaDeviceProp::maxSurfaceCubemapLayered[2])
			//Максимальные размеры слоистых Cubemap поверхности

			//printf(":  %d\n", prop.maxTexture1D)
			//Максимальный размер текстуры 1D

			//ИНТ cudaDeviceProp::maxTexture1DLayered[2])
			//Максимальные размеры 1D слоистых текстуры

			//printf(":  %d\n", prop.maxTexture1DLinear)
			//Максимальный размер 1D текстур связаны с линейной памятью

			//printf(":  %d\n", prop.maxTexture1DMipmap)
			//Максимальный 1D мип размер текстуры

			//ИНТ cudaDeviceProp::maxTexture2D[2])
			//Максимальные размеры 2D текстуры

			//ИНТ cudaDeviceProp::maxTexture2DGather[2])
			//Максимальные размеры 2D текстуры, если текстура собирайте операции должны быть выполнены

			//ИНТ cudaDeviceProp::maxTexture2DLayered[3])
			//Максимальные размеры 2D слоистых текстуры

			//ИНТ cudaDeviceProp::maxTexture2DLinear[3])
			//Максимальные размеры(ширина, высота, тангаж) для 2D текстур, связанных с скатной памятью

			//ИНТ cudaDeviceProp::maxTexture2DMipmap[2])
			//Максимальные размеры 2D мип текстуры

			//ИНТ cudaDeviceProp::maxTexture3D[3])
			//Максимальные размеры текстуры 3D

			//ИНТ cudaDeviceProp::maxTexture3DAlt[3])
			//Максимальные размеры чередуются 3D текстуры

			//printf(":  %d\n", prop.maxTextureCubemap)
			//Максимальные размеры текстуры Cubemap

			//ИНТ cudaDeviceProp::maxTextureCubemapLayered[2])
			//Максимальные размеры Cubemap слоистых текстуры

			//printf(":  %d\n", prop.maxThreadsPerMultiProcessor)
			//Максимальные потоки резидентов на многопроцессорных

			//printf(":  %d\n", prop.memoryBusWidth)
			//Ширина шины Глобальной памяти в битах

			//printf(":  %d\n", prop.multiGpuBoardGroupID)
			//Уникальный идентификатор для группы устройств на одной и той же мульти - GPU платы

			//printf(":  %d\n", prop.pageableMemoryAccess)
			//Устройство поддерживает когерентно доступ к выгружаемой памяти без вызова cudaHostRegister на нем

			//printf(":  %d\n", prop.pciBusID)
			//Шина PCI идентификатор устройства

			//printf(":  %d\n", prop.pciDeviceID)
			//PCI - устройство Идентификатор устройства

			//printf(":  %d\n", prop.pciDomainID)
			//PCI - идентификатор домена устройства

			//printf(":  %d\n", prop.regsPerMultiprocessor)
			//32 - битных регистров, доступных в многопроцессорной

			//size_t cudaDeviceProp::sharedMemPerMultiprocessor)
			//Общая память, доступная на многопроцессорной в байтах

			//printf(":  %d\n", prop.singleToDoublePrecisionPerfRatio)
			//Соотношение производительности одной точности(в операциях с плавающей запятой в секунду) до двойной точности выполнения

			//printf(":  %d\n", prop.streamPrioritiesSupported)
			//Устройство поддерживает приоритеты потоков

			//size_t cudaDeviceProp::surfaceAlignment)
			//Выравнивание требования к поверхности

			//printf(":  %d\n", prop.tccDriver)
			//1, если устройство является устройством Тесла, используя драйвер TCC, 0 в противном случае

			//size_t cudaDeviceProp::texturePitchAlignment)
			//Шаг требование выравнивания для ссылок текстур связаны с скатной памятью

			//printf(":  %d\n", prop.unifiedAddressing)
			//Устройство разделяет единое адресное пространство с хозяином


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