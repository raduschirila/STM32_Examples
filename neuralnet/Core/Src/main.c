#include "main.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "string.h"

UART_HandleTypeDef huart3;
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART3_UART_Init(void);
static void MX_USB_OTG_HS_USB_Init(void);

#define activation_relu(x) (x<0 ? 0:x)
#define activation_relu_back(x) (x<0?0:1)
#define activation_sigmoid(x) (1.0/(1.0 + exp(-x)))
#define activation_sigmoid_back(x) (x*(1.0-x))


//HYPERPARAMS
static float lr=0.0000001;
int layers=3;
int batch=4;
int epochs = 120;

//PRIMITIVE GENERAL STRUCTURES
struct matrix{
    float *value;
    int row,col;
};
struct matrix inputs[4];
struct matrix expected[4];
struct network{
    struct matrix weights;
    struct matrix inputs;
    struct matrix output;
    //short activation; // 0 relu, 1 sigmoid for now
};
struct network architecture[5];

//FUNCTION DEFINITIONS START
struct matrix init_mat(int row, int col)
{
    struct matrix x;
    x.row = row;
    x.col = col;
    x.value = (float *)malloc(sizeof(float)*x.row * x.col);
    return x;
}

struct matrix matmul(struct matrix A,struct matrix B)
{
    struct matrix result;
    if(A.col != B.row)
    {
			char buffer[256] = "Issue: matmul";
			HAL_UART_Transmit(&huart3,(uint8_t *)buffer,strlen(buffer),100);
			return result;
    }
    else {
        float sum;
        //result = init_mat(A.row, B.col);
        result.row = A.row;
        result.col = B.col;
        result.value = (float *)malloc(sizeof(float)*result.row * result.col);
        for (int row = 0; row < A.row; ++row)
        {
            for (int col = 0; col < B.col; ++col)
            {
                sum = 0;
                for (int it = 0; it < B.row; ++it)
                {
                    sum += (*(A.value + A.col*row + it) * *(B.value +B.col * it + col));
                }
                *(result.value + result.col * row + col) = sum;
            }
        }
        return result;
    }
}
struct matrix matrix_add(struct matrix A, struct matrix B)
{
    struct matrix result;
    if(A.row !=B.row || A.col != B.col)
    {
			char buffer[256] = "Issue: add";
	HAL_UART_Transmit(&huart3,(uint8_t *)buffer,strlen(buffer),100);
        return result;
    }
    else
    {
        result.row = A.row;result.col = A.col;
        result.value = (float *)malloc(sizeof(float)*result.row * result.col);
        //result = init_mat(A.row, A.col);
        for(int i=0;i<=A.row*A.col;++i)
        {
            *(result.value+i) = *(A.value+i) + *(B.value+i);
        }
        return result;
    }
}
struct matrix matrix_subtract(struct matrix A, struct matrix B)
{
    struct matrix result;
    if(A.row !=B.row || A.col != B.col)
    {
char buffer[256] = "Issue: substract";
	HAL_UART_Transmit(&huart3,(uint8_t *)buffer,strlen(buffer),100);
        return result;
    }
    else
    {
        result.row = A.row;result.col = A.col;
        result.value = (float *)malloc(sizeof(float)*result.row * result.col);
        for(int i=0;i<=A.row*A.col;++i)
        {
            *(result.value+i) = *(A.value+i) - *(B.value+i);
        }
        return result;
    }
}
struct matrix matrix_transpose(struct matrix A)
{
    if(A.row == 1 || A.col == 1)
    {
        A.row ^= A.col ^= A.row ^= A.col;
        return A;
    }
    else {
        struct matrix result;
        result.row = A.row;result.col = A.col;
        result.value = (float *)malloc(sizeof(float)*result.row * result.col);
        for (int i = 0; i < A.row; ++i) {
            for (int j = 0; j < A.col; ++j) {
                *(result.value + A.row * i + j) = *(A.value + i + A.col * j);
            }
        }
        result.row ^= result.col ^= result.row ^= result.col;
        return result;
    }
}
struct matrix scalar_multiplication(struct matrix A, float x)
{
    struct matrix result;
    result.row = A.row; result.col = A.col;
    result.value = (float *)malloc(sizeof(float)*result.row * result.col);
    for(int i=0;i<=A.row*A.col;++i)
    {
        *(result.value+i) = *(A.value+i) * x;
    }
    return result;
}
struct matrix matmul_elementwise(struct matrix A, struct matrix B)
{
    struct matrix result;
    if(A.row !=B.row || A.col != B.col)
    {
char buffer[256] = "Issue: matmul elmnt";
	HAL_UART_Transmit(&huart3,(uint8_t *)buffer,strlen(buffer),100);
        return result;
    }
    else
    {
        result.row = A.row;result.col = A.col;
        result.value = (float *)malloc(sizeof(float)*result.row * result.col);
        for(int i=0;i<=A.row*A.col;++i)
        {
            *(result.value+i) = *(A.value+i) * *(B.value+i);
        }
        return result;
    }
}// Hadamard product function

struct matrix matmul_kroneker(struct matrix A, struct matrix B)
{
    struct matrix result;
    int startRow, startCol;
    result.row = A.row*B.row;
    result.col = A.col*B.col;
    result.value = (float *)malloc(sizeof(float)*result.row * result.col);
    for(int i=0;i<A.row;i++){
        for(int j=0;j<A.col;j++){
            startRow = i*B.row;
            startCol = j*B.col;
            for(int k=0;k<B.row;k++){
                for(int l=0;l<B.col;l++){
                    *(result.value + result.col*(startRow + k) + startCol + l) = *(A.value + A.col * i + j) * *(B.value + B.col*k + l);
                }
            }
        }
    }
    return result;
}
struct  matrix update_weights(struct matrix w, float lr, struct matrix delta, struct matrix output)
{
    w = matrix_add(w, scalar_multiplication(matmul_kroneker(matrix_transpose(output), delta),lr));
    return w;
}
struct matrix backprop(struct matrix exp, struct matrix output,struct matrix input)
{
    struct matrix error,delta;
    delta.row = input.row;
    delta.col = input.col;
    delta.value = (float *)malloc(sizeof(float)*delta.row * delta.col);
    for(int i=0;i<delta.row*delta.col;++i)
    {
        *(delta.value+i) = activation_relu_back(*(output.value + i));
    }
    error = matrix_subtract(exp,output);
    delta = (delta.col == 1 ? scalar_multiplication(delta,*(error.value)): matmul_elementwise(error, delta));
    return delta;
    //error = (expected - output) * transfer_derivative(output), where transfer derivative is activation_back
    //error = weight_k * error_j * transfer_derivative(output)
}// can be recursive!!!!!
struct matrix forward_propagation(struct matrix w, struct matrix x)
{
    struct matrix out;
    out = matmul(w,x);
    for(int i=0;i<=out.col*out.row;++i)
    {
        *(out.value+i) = activation_relu(*(out.value+i));
    }
    return out;
} // can be recursive

void print_error(float b)
{
	char buffer[256];
	HAL_UART_Transmit(&huart3,(uint8_t *)buffer,sprintf(buffer,"%f\n",b),100);
}
float mean_error()
{
	float current_error=0;
	for(int j=0;j<batch;++j)
        {
            //forward pass
            for(int l=0;l<layers;++l)
            {
                    architecture[l+1].inputs = forward_propagation(matrix_transpose(architecture[l].weights),(l == 0 ? inputs[j] : architecture[l].inputs));
                    architecture[l].output = architecture[l + 1].inputs;
                if(l==layers-1)
                    architecture[l].output = forward_propagation(matrix_transpose(architecture[l].weights),architecture[l].inputs);
            }
		current_error += abs(*(architecture[layers-1].output.value)-*(expected[j].value));
					}
    return current_error/4;
}
void train(int epochs)
{
    for(int i=0;i<epochs;++i)
    {
        for(int j=0;j<batch;++j)
        {
            //forward pass
            for(int l=0;l<layers;++l)
            {
                    architecture[l+1].inputs = forward_propagation(matrix_transpose(architecture[l].weights),
                                                                     (l == 0 ? inputs[j] : architecture[l].inputs));

                if(l==layers-1){
                    architecture[l].output = forward_propagation(matrix_transpose(architecture[l].weights),architecture[l].inputs);}
								else{
									architecture[l].output = architecture[l + 1].inputs;
								}
            }
            //backwards pass
            for(int l=layers-1;l>=0;--l) // goes 2 1 0 2 not updated
            {
                architecture[l].weights = update_weights(architecture[l].weights, lr,
                               backprop((l==layers-1?expected[j]:architecture[l].output), architecture[l].output,
                                        (l==0?inputs[j]:architecture[l].inputs)), architecture[l].output);
            }
        }
        print_error(mean_error());
    }
}
void randomize_weights()
{
    for(int i=0;i<layers;++i)
    {
        for(int j=0;j<architecture[i].weights.row*architecture[i].weights.col;++j)
        {
            *(architecture[i].weights.value+j)= 1.3;//((float) rand()/(float)(RAND_MAX)*10.0);
        }
    }
}
void init_architecture()
{

    layers=3;
    int x=6,y=6,z=1;

	architecture[0].weights = init_mat(x,x);
	architecture[0].output= init_mat(x,1);
	architecture[1].weights = init_mat(x,y);
	architecture[1].output = init_mat(y,1);
	architecture[2].weights = init_mat(y,z);
	architecture[2].output = init_mat(z,1);

    randomize_weights();

    inputs[0].row = 6;
    inputs[0].col = 1;

    for(int b=0;b<batch;++b) {
        inputs[b].row = inputs[0].row;
        inputs[b].col = inputs[0].col;
        inputs[b].value = (float *) malloc(sizeof(float) * inputs[0].row * inputs[0].col);
    }
    for (int b=0;b<batch;++b){
        for (int i = 0; i < inputs[b].row * inputs[b].col; ++i) {
            *(inputs[b].value + i)=1.0-0.5*b;
        }
    }

        expected[0].row = 1;
        expected[0].col = 1;
    for(int b=0;b<batch;++b) {
        expected[b].row = expected[0].row;
        expected[b].col = expected[0].col;
        expected[b].value = (float *) malloc(sizeof(float) * expected[0].row * expected[0].col);
    }
    for(int b=0;b<batch;++b) {
        for (int i = 0; i < expected[b].row * expected[b].col; ++i) {
           *(expected[b].value + i)=1-.5*b;
        }
    }
}

int main(void)
{

  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();
  MX_USART3_UART_Init();
  MX_USB_OTG_HS_USB_Init();
  init_architecture();
  train(epochs);
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Supply configuration update enable
  */
  HAL_PWREx_ConfigSupply(PWR_DIRECT_SMPS_SUPPLY);

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE0);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI48|RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_BYPASS;
  RCC_OscInitStruct.HSI48State = RCC_HSI48_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 1;
  RCC_OscInitStruct.PLL.PLLN = 16;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_3;
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_D3PCLK1|RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV1;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_1) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief USART3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART3_UART_Init(void)
{

  /* USER CODE BEGIN USART3_Init 0 */

  /* USER CODE END USART3_Init 0 */

  /* USER CODE BEGIN USART3_Init 1 */

  /* USER CODE END USART3_Init 1 */
  huart3.Instance = USART3;
  huart3.Init.BaudRate = 115200;
  huart3.Init.WordLength = UART_WORDLENGTH_8B;
  huart3.Init.StopBits = UART_STOPBITS_1;
  huart3.Init.Parity = UART_PARITY_NONE;
  huart3.Init.Mode = UART_MODE_TX_RX;
  huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart3.Init.OverSampling = UART_OVERSAMPLING_16;
  huart3.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart3.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart3.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart3, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart3, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART3_Init 2 */

  /* USER CODE END USART3_Init 2 */

}

/**
  * @brief USB_OTG_HS Initialization Function
  * @param None
  * @retval None
  */
static void MX_USB_OTG_HS_USB_Init(void)
{

  /* USER CODE BEGIN USB_OTG_HS_Init 0 */

  /* USER CODE END USB_OTG_HS_Init 0 */

  /* USER CODE BEGIN USB_OTG_HS_Init 1 */

  /* USER CODE END USB_OTG_HS_Init 1 */
  /* USER CODE BEGIN USB_OTG_HS_Init 2 */

  /* USER CODE END USB_OTG_HS_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOF_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOG_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(USB_FS_PWR_EN_GPIO_Port, USB_FS_PWR_EN_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, LD1_Pin|LD3_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : B1_Pin */
  GPIO_InitStruct.Pin = B1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : USB_FS_PWR_EN_Pin */
  GPIO_InitStruct.Pin = USB_FS_PWR_EN_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(USB_FS_PWR_EN_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : LD1_Pin LD3_Pin */
  GPIO_InitStruct.Pin = LD1_Pin|LD3_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pin : USB_FS_OVCR_Pin */
  GPIO_InitStruct.Pin = USB_FS_OVCR_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(USB_FS_OVCR_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : USB_FS_VBUS_Pin */
  GPIO_InitStruct.Pin = USB_FS_VBUS_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(USB_FS_VBUS_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : USB_FS_ID_Pin */
  GPIO_InitStruct.Pin = USB_FS_ID_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF10_OTG1_HS;
  HAL_GPIO_Init(USB_FS_ID_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : USB_FS_N_Pin USB_FS_P_Pin */
  GPIO_InitStruct.Pin = USB_FS_N_Pin|USB_FS_P_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pin : LD2_Pin */
  GPIO_InitStruct.Pin = LD2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LD2_GPIO_Port, &GPIO_InitStruct);

}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
