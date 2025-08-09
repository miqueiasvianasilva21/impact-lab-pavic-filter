#pragma once

#include <thread>
#include <algorithm>
#include <vector> 
#include <msclr/gcroot.h>
#include <chrono>
#include "CudaFilter.h" 

namespace pavicgui2024 {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;
	using namespace System::Drawing::Imaging;
	using namespace std::chrono;
	using namespace std;

	// Funções auxiliares
	void ApplySepiaFilterPartial(Bitmap^ inputImage, Bitmap^ outputImage, int startY, int endY) {
		for (int i = 0; i < inputImage->Width; i++) {
			for (int j = startY; j < endY; j++) {
				Color pixelColor = inputImage->GetPixel(i, j);
				int r = pixelColor.R;
				int g = pixelColor.G;
				int b = pixelColor.B;
				double tr = 0.393 * r + 0.769 * g + 0.189 * b;
				double tg = 0.349 * r + 0.686 * g + 0.168 * b;
				double tb = 0.272 * r + 0.534 * g + 0.131 * b;
				int newR = Math::Min(255, (int)tr);
				int newG = Math::Min(255, (int)tg);
				int newB = Math::Min(255, (int)tb);
				outputImage->SetPixel(i, j, Color::FromArgb(newR, newG, newB));
			}
		}
	}
	void ApplyFilterRegion_Raw(unsigned char* inPtr, unsigned char* outPtr, int stride, int startY, int endY, int width) {
		for (int y = startY; y < endY; y++) {
			for (int x = 0; x < width; x++) {
				long offset = (long)y * stride + (long)x * 3;
				float b = inPtr[offset];
				float g = inPtr[offset + 1];
				float r = inPtr[offset + 2];
				float new_r = r * 0.393f + g * 0.769f + b * 0.189f;
				float new_g = r * 0.349f + g * 0.686f + b * 0.168f;
				float new_b = r * 0.272f + g * 0.534f + b * 0.131f;
				auto clamp = [](float val) { return (unsigned char)Math::Min(255.0f, val); };
				outPtr[offset] = clamp(new_b);
				outPtr[offset + 1] = clamp(new_g);
				outPtr[offset + 2] = clamp(new_r);
			}
		}
	}
	void ApplySepiaFilterWindow(Bitmap^ inputImage, Bitmap^ outputImage, int startX, int endX, int startY, int endY) {
		for (int i = startX; i < endX; i++) {
			for (int j = startY; j < endY; j++) {
				Color pixelColor = inputImage->GetPixel(i, j);
				int r = pixelColor.R;
				int g = pixelColor.G;
				int b = pixelColor.B;
				double tr = 0.393 * r + 0.769 * g + 0.189 * b;
				double tg = 0.349 * r + 0.686 * g + 0.168 * b;
				double tb = 0.272 * r + 0.534 * g + 0.131 * b;
				int newR = Math::Min(255, (int)tr);
				int newG = Math::Min(255, (int)tg);
				int newB = Math::Min(255, (int)tb);
				outputImage->SetPixel(i, j, Color::FromArgb(newR, newG, newB));
			}
		}
	}

	public ref class Pavic_gui_2024_Form : public System::Windows::Forms::Form
	{
	public:
		Pavic_gui_2024_Form(void)
		{
			InitializeComponent();
		}

	protected:
		~Pavic_gui_2024_Form()
		{
			if (components)
			{
				delete components;
			}
		}

	private: System::Windows::Forms::Button^ bt_open;
	private: System::Windows::Forms::Button^ bt_close;
	private: System::Windows::Forms::Button^ bt_exit;
	private: System::Windows::Forms::PictureBox^ pbox_input;
	private: System::Windows::Forms::PictureBox^ pbox_copy;
	private: System::Windows::Forms::PictureBox^ pbox_output;
	private: System::Windows::Forms::Button^ bt_copy;
	private: System::Windows::Forms::Button^ bt_filter_bw;
	private: System::Windows::Forms::Label^ label1;
	private: System::Windows::Forms::Label^ label2;
	private: System::Windows::Forms::Button^ bt_close_copy;
	private: System::Windows::Forms::Button^ bt_close_output;
	private: System::Windows::Forms::Button^ bt_filter_Sepia;
	private: System::Windows::Forms::Button^ bt_filter_Sepia_MultiThread;
	private: System::Windows::Forms::Button^ bt_filter_Sepia_top;
	private: System::Windows::Forms::Button^ bt_filter_Sepia_botton;
	private: System::Windows::Forms::Label^ lb_timer;
	private: System::Windows::Forms::TextBox^ textB_Time;
	private: System::Windows::Forms::Button^ button1;
	private: System::Windows::Forms::Button^ bt_filter_Sepia_left;
	private: System::Windows::Forms::Button^ bt_filter_Sepia_Thread;
	private: System::Windows::Forms::Button^ bt_filter_cuda;
	private: System::Windows::Forms::Button^ bt_filter_invert;

	private:
		System::ComponentModel::Container^ components;

#pragma region Windows Form Designer generated code
		void InitializeComponent(void)
		{
			this->bt_open = (gcnew System::Windows::Forms::Button());
			this->bt_close = (gcnew System::Windows::Forms::Button());
			this->bt_exit = (gcnew System::Windows::Forms::Button());
			this->pbox_input = (gcnew System::Windows::Forms::PictureBox());
			this->pbox_copy = (gcnew System::Windows::Forms::PictureBox());
			this->pbox_output = (gcnew System::Windows::Forms::PictureBox());
			this->bt_copy = (gcnew System::Windows::Forms::Button());
			this->bt_filter_bw = (gcnew System::Windows::Forms::Button());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->bt_close_copy = (gcnew System::Windows::Forms::Button());
			this->bt_close_output = (gcnew System::Windows::Forms::Button());
			this->bt_filter_Sepia = (gcnew System::Windows::Forms::Button());
			this->bt_filter_Sepia_MultiThread = (gcnew System::Windows::Forms::Button());
			this->bt_filter_Sepia_top = (gcnew System::Windows::Forms::Button());
			this->bt_filter_Sepia_botton = (gcnew System::Windows::Forms::Button());
			this->lb_timer = (gcnew System::Windows::Forms::Label());
			this->textB_Time = (gcnew System::Windows::Forms::TextBox());
			this->button1 = (gcnew System::Windows::Forms::Button());
			this->bt_filter_Sepia_left = (gcnew System::Windows::Forms::Button());
			this->bt_filter_Sepia_Thread = (gcnew System::Windows::Forms::Button());
			this->bt_filter_cuda = (gcnew System::Windows::Forms::Button());
			this->bt_filter_invert = (gcnew System::Windows::Forms::Button());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pbox_input))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pbox_copy))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pbox_output))->BeginInit();
			this->SuspendLayout();
			// 
			// bt_open
			// 
			this->bt_open->Location = System::Drawing::Point(12, 12);
			this->bt_open->Name = L"bt_open";
			this->bt_open->Size = System::Drawing::Size(189, 46);
			this->bt_open->TabIndex = 0;
			this->bt_open->Text = L"Abrir";
			this->bt_open->UseVisualStyleBackColor = true;
			this->bt_open->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::bt_open_Click);
			// 
			// bt_close
			// 
			this->bt_close->Location = System::Drawing::Point(388, 236);
			this->bt_close->Name = L"bt_close";
			this->bt_close->Size = System::Drawing::Size(127, 34);
			this->bt_close->TabIndex = 1;
			this->bt_close->Text = L"Fechar";
			this->bt_close->UseVisualStyleBackColor = true;
			this->bt_close->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::bt_close_Click);
			// 
			// bt_exit
			// 
			this->bt_exit->Location = System::Drawing::Point(12, 114);
			this->bt_exit->Name = L"bt_exit";
			this->bt_exit->Size = System::Drawing::Size(189, 46);
			this->bt_exit->TabIndex = 2;
			this->bt_exit->Text = L"Sair";
			this->bt_exit->UseVisualStyleBackColor = true;
			this->bt_exit->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::bt_exit_Click);
			// 
			// pbox_input
			// 
			this->pbox_input->BorderStyle = System::Windows::Forms::BorderStyle::Fixed3D;
			this->pbox_input->Location = System::Drawing::Point(17, 277);
			this->pbox_input->Name = L"pbox_input";
			this->pbox_input->Size = System::Drawing::Size(497, 406);
			this->pbox_input->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pbox_input->TabIndex = 3;
			this->pbox_input->TabStop = false;
			// 
			// pbox_copy
			// 
			this->pbox_copy->BorderStyle = System::Windows::Forms::BorderStyle::Fixed3D;
			this->pbox_copy->Location = System::Drawing::Point(555, 277);
			this->pbox_copy->Name = L"pbox_copy";
			this->pbox_copy->Size = System::Drawing::Size(497, 406);
			this->pbox_copy->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pbox_copy->TabIndex = 4;
			this->pbox_copy->TabStop = false;
			// 
			// pbox_output
			// 
			this->pbox_output->BorderStyle = System::Windows::Forms::BorderStyle::Fixed3D;
			this->pbox_output->Location = System::Drawing::Point(1087, 277);
			this->pbox_output->Name = L"pbox_output";
			this->pbox_output->Size = System::Drawing::Size(497, 406);
			this->pbox_output->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pbox_output->TabIndex = 5;
			this->pbox_output->TabStop = false;
			// 
			// bt_copy
			// 
			this->bt_copy->Location = System::Drawing::Point(12, 63);
			this->bt_copy->Name = L"bt_copy";
			this->bt_copy->Size = System::Drawing::Size(189, 46);
			this->bt_copy->TabIndex = 6;
			this->bt_copy->Text = L"Copiar";
			this->bt_copy->UseVisualStyleBackColor = true;
			this->bt_copy->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::bt_copy_Click);
			// 
			// bt_filter_bw
			// 
			this->bt_filter_bw->Location = System::Drawing::Point(288, 14);
			this->bt_filter_bw->Name = L"bt_filter_bw";
			this->bt_filter_bw->Size = System::Drawing::Size(189, 46);
			this->bt_filter_bw->TabIndex = 7;
			this->bt_filter_bw->Text = L"Filtro P&B";
			this->bt_filter_bw->UseVisualStyleBackColor = true;
			this->bt_filter_bw->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::bt_filter_bw_Click);
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(1437, 709);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(230, 16);
			this->label1->TabIndex = 8;
			this->label1->Text = L" Miquéias Viana Silva e Romulo Torres";
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->Location = System::Drawing::Point(9, 709);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(111, 16);
			this->label2->TabIndex = 9;
			this->label2->Text = L" PAVIC LAB: 2025";
			// 
			// bt_close_copy
			// 
			this->bt_close_copy->Location = System::Drawing::Point(925, 236);
			this->bt_close_copy->Name = L"bt_close_copy";
			this->bt_close_copy->Size = System::Drawing::Size(127, 34);
			this->bt_close_copy->TabIndex = 10;
			this->bt_close_copy->Text = L"Fechar";
			this->bt_close_copy->UseVisualStyleBackColor = true;
			this->bt_close_copy->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::bt_close_copy_Click);
			// 
			// bt_close_output
			// 
			this->bt_close_output->Location = System::Drawing::Point(1459, 236);
			this->bt_close_output->Name = L"bt_close_output";
			this->bt_close_output->Size = System::Drawing::Size(127, 34);
			this->bt_close_output->TabIndex = 11;
			this->bt_close_output->Text = L"Fechar";
			this->bt_close_output->UseVisualStyleBackColor = true;
			this->bt_close_output->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::bt_close_output_Click);
			// 
			// bt_filter_Sepia
			// 
			this->bt_filter_Sepia->Location = System::Drawing::Point(288, 64);
			this->bt_filter_Sepia->Name = L"bt_filter_Sepia";
			this->bt_filter_Sepia->Size = System::Drawing::Size(189, 46);
			this->bt_filter_Sepia->TabIndex = 12;
			this->bt_filter_Sepia->Text = L"Filtro Sépia (CPU)";
			this->bt_filter_Sepia->UseVisualStyleBackColor = true;
			this->bt_filter_Sepia->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::bt_filter_Sepia_Click);
			// 
			// bt_filter_Sepia_MultiThread
			// 
			this->bt_filter_Sepia_MultiThread->Location = System::Drawing::Point(288, 114);
			this->bt_filter_Sepia_MultiThread->Name = L"bt_filter_Sepia_MultiThread";
			this->bt_filter_Sepia_MultiThread->Size = System::Drawing::Size(189, 46);
			this->bt_filter_Sepia_MultiThread->TabIndex = 13;
			this->bt_filter_Sepia_MultiThread->Text = L"Sépia (CPU Multi-Thread)";
			this->bt_filter_Sepia_MultiThread->UseVisualStyleBackColor = true;
			this->bt_filter_Sepia_MultiThread->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::bt_filter_Sepia_MultiThread_Click);
			// 
			// bt_filter_Sepia_top
			// 
			this->bt_filter_Sepia_top->Location = System::Drawing::Point(483, 14);
			this->bt_filter_Sepia_top->Name = L"bt_filter_Sepia_top";
			this->bt_filter_Sepia_top->Size = System::Drawing::Size(189, 46);
			this->bt_filter_Sepia_top->TabIndex = 14;
			this->bt_filter_Sepia_top->Text = L"Sépia - Cima";
			this->bt_filter_Sepia_top->UseVisualStyleBackColor = true;
			this->bt_filter_Sepia_top->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::bt_filter_Sepia_top_Click);
			// 
			// bt_filter_Sepia_botton
			// 
			this->bt_filter_Sepia_botton->Location = System::Drawing::Point(677, 14);
			this->bt_filter_Sepia_botton->Name = L"bt_filter_Sepia_botton";
			this->bt_filter_Sepia_botton->Size = System::Drawing::Size(189, 46);
			this->bt_filter_Sepia_botton->TabIndex = 15;
			this->bt_filter_Sepia_botton->Text = L"Sépia - Baixo";
			this->bt_filter_Sepia_botton->UseVisualStyleBackColor = true;
			this->bt_filter_Sepia_botton->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::bt_filter_Sepia_botton_Click);
			// 
			// lb_timer
			// 
			this->lb_timer->AutoSize = true;
			this->lb_timer->Location = System::Drawing::Point(985, 92);
			this->lb_timer->Name = L"lb_timer";
			this->lb_timer->Size = System::Drawing::Size(42, 16);
			this->lb_timer->TabIndex = 16;
			this->lb_timer->Text = L"Timer";
			// 
			// textB_Time
			// 
			this->textB_Time->Location = System::Drawing::Point(989, 112);
			this->textB_Time->Name = L"textB_Time";
			this->textB_Time->Size = System::Drawing::Size(213, 22);
			this->textB_Time->TabIndex = 17;
			// 
			// button1
			// 
			this->button1->Location = System::Drawing::Point(677, 65);
			this->button1->Name = L"button1";
			this->button1->Size = System::Drawing::Size(189, 46);
			this->button1->TabIndex = 19;
			this->button1->Text = L"Sépia - Direita";
			this->button1->UseVisualStyleBackColor = true;
			this->button1->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::bt_filter_Sepia_right_Click);
			// 
			// bt_filter_Sepia_left
			// 
			this->bt_filter_Sepia_left->Location = System::Drawing::Point(483, 65);
			this->bt_filter_Sepia_left->Name = L"bt_filter_Sepia_left";
			this->bt_filter_Sepia_left->Size = System::Drawing::Size(189, 46);
			this->bt_filter_Sepia_left->TabIndex = 18;
			this->bt_filter_Sepia_left->Text = L"Sépia - Esquerda";
			this->bt_filter_Sepia_left->UseVisualStyleBackColor = true;
			this->bt_filter_Sepia_left->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::bt_filter_Sepia_left_Click);
			// 
			// bt_filter_Sepia_Thread
			// 
			this->bt_filter_Sepia_Thread->Location = System::Drawing::Point(483, 116);
			this->bt_filter_Sepia_Thread->Name = L"bt_filter_Sepia_Thread";
			this->bt_filter_Sepia_Thread->Size = System::Drawing::Size(189, 46);
			this->bt_filter_Sepia_Thread->TabIndex = 20;
			this->bt_filter_Sepia_Thread->Text = L"Sépia - Thread (Teste)";
			this->bt_filter_Sepia_Thread->UseVisualStyleBackColor = true;
			this->bt_filter_Sepia_Thread->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::bt_filter_Sepia_Thread_Click);
			// 
			// bt_filter_cuda
			// 
			this->bt_filter_cuda->Location = System::Drawing::Point(677, 116);
			this->bt_filter_cuda->Name = L"bt_filter_cuda";
			this->bt_filter_cuda->Size = System::Drawing::Size(189, 46);
			this->bt_filter_cuda->TabIndex = 21;
			this->bt_filter_cuda->Text = L"Sépia (CUDA)";
			this->bt_filter_cuda->UseVisualStyleBackColor = true;
			this->bt_filter_cuda->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::bt_filter_cuda_Click);
			// 
			// bt_filter_invert
			// 
			this->bt_filter_invert->Location = System::Drawing::Point(288, 164);
			this->bt_filter_invert->Name = L"bt_filter_invert";
			this->bt_filter_invert->Size = System::Drawing::Size(189, 46);
			this->bt_filter_invert->TabIndex = 22;
			this->bt_filter_invert->Text = L"Inverter (CUDA)";
			this->bt_filter_invert->UseVisualStyleBackColor = true;
			this->bt_filter_invert->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::bt_filter_invert_Click);
			// 
			// Pavic_gui_2024_Form
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(8, 16);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(1668, 759);
			this->Controls->Add(this->bt_filter_invert);
			this->Controls->Add(this->bt_filter_cuda);
			this->Controls->Add(this->bt_filter_Sepia_Thread);
			this->Controls->Add(this->button1);
			this->Controls->Add(this->bt_filter_Sepia_left);
			this->Controls->Add(this->textB_Time);
			this->Controls->Add(this->lb_timer);
			this->Controls->Add(this->bt_filter_Sepia_botton);
			this->Controls->Add(this->bt_filter_Sepia_top);
			this->Controls->Add(this->bt_filter_Sepia_MultiThread);
			this->Controls->Add(this->bt_filter_Sepia);
			this->Controls->Add(this->bt_close_output);
			this->Controls->Add(this->bt_close_copy);
			this->Controls->Add(this->label2);
			this->Controls->Add(this->label1);
			this->Controls->Add(this->bt_filter_bw);
			this->Controls->Add(this->bt_copy);
			this->Controls->Add(this->pbox_output);
			this->Controls->Add(this->pbox_copy);
			this->Controls->Add(this->pbox_input);
			this->Controls->Add(this->bt_exit);
			this->Controls->Add(this->bt_close);
			this->Controls->Add(this->bt_open);
			this->Name = L"Pavic_gui_2024_Form";
			this->Text = L"PROJECT: IMPACTLAB LAB 2025";
			this->Load += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::Pavic_gui_2024_Form_Load);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pbox_input))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pbox_copy))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pbox_output))->EndInit();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion

	private: System::Void bt_open_Click(System::Object^ sender, System::EventArgs^ e) {
		OpenFileDialog^ ofd = gcnew OpenFileDialog();
		if (ofd->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
			pbox_input->ImageLocation = ofd->FileName;
		}
	}
	private: System::Void bt_close_Click(System::Object^ sender, System::EventArgs^ e) {
		pbox_input->Image = nullptr;
	}
	private: System::Void bt_copy_Click(System::Object^ sender, System::EventArgs^ e) {
		if (pbox_input->Image == nullptr) return;
		pbox_copy->Image = pbox_input->Image;
		lb_timer->Text = "Operação de Cópia";
		textB_Time->Text = "N/A";
	}
	private: System::Void bt_filter_bw_Click(System::Object^ sender, System::EventArgs^ e) {
		if (pbox_input->Image == nullptr) return;

		auto start = high_resolution_clock::now();

		Bitmap^ inputImage = (Bitmap^)pbox_input->Image;
		Bitmap^ outputImage = gcnew Bitmap(inputImage->Width, inputImage->Height);
		for (int i = 0; i < inputImage->Width; i++) {
			for (int j = 0; j < inputImage->Height; j++) {
				Color pixelColor = inputImage->GetPixel(i, j);
				int grayValue = (int)(0.299 * pixelColor.R + 0.587 * pixelColor.G + 0.114 * pixelColor.B);
				outputImage->SetPixel(i, j, Color::FromArgb(grayValue, grayValue, grayValue));
			}
		}

		auto end = high_resolution_clock::now();
		auto duration = duration_cast<milliseconds>(end - start);
		lb_timer->Text = "Filtro P&B (CPU):";
		textB_Time->Text = duration.count().ToString() + " ms";

		pbox_output->Image = outputImage;
	}
	private: System::Void bt_exit_Click(System::Object^ sender, System::EventArgs^ e) {
		Application::Exit();
	}
	private: System::Void bt_close_copy_Click(System::Object^ sender, System::EventArgs^ e) {
		pbox_copy->Image = nullptr;
	}
	private: System::Void bt_close_output_Click(System::Object^ sender, System::EventArgs^ e) {
		pbox_output->Image = nullptr;
	}
	private: System::Void Pavic_gui_2024_Form_Load(System::Object^ sender, System::EventArgs^ e) {
	}
	private: System::Void bt_filter_Sepia_Click(System::Object^ sender, System::EventArgs^ e) {
		if (pbox_input->Image == nullptr) return;

		auto start = high_resolution_clock::now();

		Bitmap^ inputImage = (Bitmap^)pbox_input->Image;
		Bitmap^ outputImage = gcnew Bitmap(inputImage->Width, inputImage->Height);
		ApplySepiaFilterWindow(inputImage, outputImage, 0, inputImage->Width, 0, inputImage->Height);

		auto end = high_resolution_clock::now();
		auto duration = duration_cast<milliseconds>(end - start);
		lb_timer->Text = "Sépia (CPU):";
		textB_Time->Text = duration.count().ToString() + " ms";

		pbox_output->Image = outputImage;
	}
	private: System::Void bt_filter_Sepia_MultiThread_Click(System::Object^ sender, System::EventArgs^ e) {
		if (pbox_input->Image == nullptr) {
			MessageBox::Show("Por favor, carregue uma imagem primeiro.", "Erro");
			return;
		}

		auto start = high_resolution_clock::now();

		Bitmap^ originalBitmap = gcnew Bitmap(pbox_input->Image);
		Bitmap^ inputBitmap = gcnew Bitmap(originalBitmap->Width, originalBitmap->Height, Imaging::PixelFormat::Format24bppRgb);
		Graphics^ g = Graphics::FromImage(inputBitmap);
		g->DrawImage(originalBitmap, 0, 0, originalBitmap->Width, originalBitmap->Height);
		delete g;
		delete originalBitmap;

		Bitmap^ outputBitmap = gcnew Bitmap(inputBitmap->Width, inputBitmap->Height, inputBitmap->PixelFormat);
		Rectangle rect = Rectangle(0, 0, inputBitmap->Width, inputBitmap->Height);
		Imaging::BitmapData^ bmpDataInput = nullptr;
		Imaging::BitmapData^ bmpDataOutput = nullptr;

		try {
			bmpDataInput = inputBitmap->LockBits(rect, Imaging::ImageLockMode::ReadOnly, inputBitmap->PixelFormat);
			bmpDataOutput = outputBitmap->LockBits(rect, Imaging::ImageLockMode::WriteOnly, outputBitmap->PixelFormat);
			unsigned char* ptrInput = (unsigned char*)bmpDataInput->Scan0.ToPointer();
			unsigned char* ptrOutput = (unsigned char*)bmpDataOutput->Scan0.ToPointer();
			int width = inputBitmap->Width;
			int height = inputBitmap->Height;
			int stride = bmpDataInput->Stride;
			const int numThreads = Environment::ProcessorCount;
			std::vector<std::thread> threads;
			int rowsPerThread = height / numThreads;

			for (int i = 0; i < numThreads; ++i) {
				int startY = i * rowsPerThread;
				int endY = (i == numThreads - 1) ? height : startY + rowsPerThread;
				threads.emplace_back(ApplyFilterRegion_Raw, ptrInput, ptrOutput, stride, startY, endY, width);
			}

			for (auto& t : threads) {
				if (t.joinable()) {
					t.join();
				}
			}
		}
		finally {
			if (bmpDataInput != nullptr) inputBitmap->UnlockBits(bmpDataInput);
			if (bmpDataOutput != nullptr) outputBitmap->UnlockBits(bmpDataOutput);
		}

		auto end = high_resolution_clock::now();
		auto duration = duration_cast<milliseconds>(end - start);
		lb_timer->Text = "Sépia (CPU Multi-Thread):";
		textB_Time->Text = duration.count().ToString() + " ms";

		pbox_output->Image = outputBitmap;
		delete inputBitmap;
	}
	private: System::Void bt_filter_Sepia_botton_Click(System::Object^ sender, System::EventArgs^ e) {
		if (pbox_input->Image == nullptr) return;

		auto start = high_resolution_clock::now();

		Bitmap^ inputImage = (Bitmap^)pbox_input->Image;
		Bitmap^ outputImage = gcnew Bitmap(inputImage->Width, inputImage->Height);
		int startY_Botton = inputImage->Height / 2;
		int endY_Botton = inputImage->Height;
		ApplySepiaFilterPartial(inputImage, outputImage, startY_Botton, endY_Botton);

		auto end = high_resolution_clock::now();
		auto duration = duration_cast<milliseconds>(end - start);
		lb_timer->Text = "Sépia (Metade de Baixo):";
		textB_Time->Text = duration.count().ToString() + " ms";

		pbox_output->Image = outputImage;
	}
	private: System::Void bt_filter_Sepia_top_Click(System::Object^ sender, System::EventArgs^ e) {
		if (pbox_input->Image == nullptr) return;

		auto start = high_resolution_clock::now();

		Bitmap^ inputImage = (Bitmap^)pbox_input->Image;
		Bitmap^ outputImage = gcnew Bitmap(inputImage->Width, inputImage->Height);
		int startY_Top = 0;
		int endY_Top = inputImage->Height / 2;
		ApplySepiaFilterPartial(inputImage, outputImage, startY_Top, endY_Top);

		auto end = high_resolution_clock::now();
		auto duration = duration_cast<milliseconds>(end - start);
		lb_timer->Text = "Sépia (Metade de Cima):";
		textB_Time->Text = duration.count().ToString() + " ms";

		pbox_copy->Image = outputImage;
	}
	private: System::Void bt_filter_Sepia_left_Click(System::Object^ sender, System::EventArgs^ e) {
		if (pbox_input->Image == nullptr) return;

		auto start = high_resolution_clock::now();

		Bitmap^ inputImage = (Bitmap^)pbox_input->Image;
		Bitmap^ outputImage = gcnew Bitmap(inputImage->Width, inputImage->Height);
		int startY_Top = 0;
		int endY_Top = inputImage->Height;
		int startX_left = 0;
		int endX_left = inputImage->Width / 2;
		ApplySepiaFilterWindow(inputImage, outputImage, startX_left, endX_left, startY_Top, endY_Top);

		auto end = high_resolution_clock::now();
		auto duration = duration_cast<milliseconds>(end - start);
		lb_timer->Text = "Sépia (Metade Esquerda):";
		textB_Time->Text = duration.count().ToString() + " ms";

		pbox_copy->Image = outputImage;
	}
	private: System::Void bt_filter_Sepia_right_Click(System::Object^ sender, System::EventArgs^ e) {
		if (pbox_input->Image == nullptr) return;

		auto start = high_resolution_clock::now();

		Bitmap^ inputImage = (Bitmap^)pbox_input->Image;
		Bitmap^ outputImage = gcnew Bitmap(inputImage->Width, inputImage->Height);
		int startY_Botton = 0;
		int endY_Botton = inputImage->Height;
		int startX_Right = inputImage->Width / 2;
		int endX_Right = inputImage->Width;
		ApplySepiaFilterWindow(inputImage, outputImage, startX_Right, endX_Right, startY_Botton, endY_Botton);

		auto end = high_resolution_clock::now();
		auto duration = duration_cast<milliseconds>(end - start);
		lb_timer->Text = "Sépia (Metade Direita):";
		textB_Time->Text = duration.count().ToString() + " ms";

		pbox_output->Image = outputImage;
	}
	private: System::Void bt_filter_Sepia_Thread_Click(System::Object^ sender, System::EventArgs^ e) {
		if (pbox_input->Image == nullptr) return;

		auto start = high_resolution_clock::now();

		Bitmap^ inputImage = (Bitmap^)pbox_input->Image;
		Bitmap^ outputImage = gcnew Bitmap(inputImage->Width, inputImage->Height);

		ApplySepiaFilterWindow(inputImage, outputImage, 0, inputImage->Width, 0, inputImage->Height);

		auto end = high_resolution_clock::now();
		auto duration = duration_cast<milliseconds>(end - start);
		lb_timer->Text = "Sépia (CPU - Botão Teste):";
		textB_Time->Text = duration.count().ToString() + " ms";

		pbox_output->Image = outputImage;
	}

	private: System::Void bt_filter_cuda_Click(System::Object^ sender, System::EventArgs^ e) {
		if (pbox_input->Image == nullptr) {
			MessageBox::Show("Por favor, carregue uma imagem primeiro.", "Erro", MessageBoxButtons::OK, MessageBoxIcon::Error);
			return;
		}

		auto start = high_resolution_clock::now();

		Bitmap^ originalBitmap = gcnew Bitmap(pbox_input->Image);
		Bitmap^ inputBitmap = gcnew Bitmap(originalBitmap->Width, originalBitmap->Height, Imaging::PixelFormat::Format24bppRgb);
		Graphics^ g = Graphics::FromImage(inputBitmap);
		g->DrawImage(originalBitmap, 0, 0, originalBitmap->Width, originalBitmap->Height);
		delete g;
		delete originalBitmap;

		Bitmap^ outputBitmap = gcnew Bitmap(inputBitmap->Width, inputBitmap->Height, inputBitmap->PixelFormat);
		Rectangle rect = Rectangle(0, 0, inputBitmap->Width, inputBitmap->Height);
		Imaging::BitmapData^ bmpDataInput = nullptr;
		Imaging::BitmapData^ bmpDataOutput = nullptr;

		try
		{
			bmpDataInput = inputBitmap->LockBits(rect, Imaging::ImageLockMode::ReadOnly, inputBitmap->PixelFormat);
			bmpDataOutput = outputBitmap->LockBits(rect, Imaging::ImageLockMode::WriteOnly, outputBitmap->PixelFormat);
			unsigned char* ptrInput = (unsigned char*)bmpDataInput->Scan0.ToPointer();
			unsigned char* ptrOutput = (unsigned char*)bmpDataOutput->Scan0.ToPointer();
			int width = inputBitmap->Width;
			int height = inputBitmap->Height;
			int stride = bmpDataInput->Stride;

			run_sepia_filter_cuda(ptrOutput, ptrInput, width, height, stride);
		}
		finally
		{
			if (bmpDataInput != nullptr) inputBitmap->UnlockBits(bmpDataInput);
			if (bmpDataOutput != nullptr) outputBitmap->UnlockBits(bmpDataOutput);
		}

		auto end = high_resolution_clock::now();
		auto duration = duration_cast<milliseconds>(end - start);
		lb_timer->Text = "Sépia (CUDA):";
		textB_Time->Text = duration.count().ToString() + " ms";

		pbox_output->Image = outputBitmap;
		delete inputBitmap;
	}

	private: System::Void bt_filter_invert_Click(System::Object^ sender, System::EventArgs^ e) {
		if (pbox_input->Image == nullptr) {
			MessageBox::Show("Por favor, carregue uma imagem primeiro.", "Erro", MessageBoxButtons::OK, MessageBoxIcon::Error);
			return;
		}

		auto start = high_resolution_clock::now();

		Bitmap^ originalBitmap = gcnew Bitmap(pbox_input->Image);
		Bitmap^ inputBitmap = gcnew Bitmap(originalBitmap->Width, originalBitmap->Height, Imaging::PixelFormat::Format24bppRgb);
		Graphics^ g = Graphics::FromImage(inputBitmap);
		g->DrawImage(originalBitmap, 0, 0, originalBitmap->Width, originalBitmap->Height);
		delete g;
		delete originalBitmap;

		Bitmap^ outputBitmap = gcnew Bitmap(inputBitmap->Width, inputBitmap->Height, inputBitmap->PixelFormat);
		Rectangle rect = Rectangle(0, 0, inputBitmap->Width, inputBitmap->Height);
		Imaging::BitmapData^ bmpDataInput = nullptr;
		Imaging::BitmapData^ bmpDataOutput = nullptr;

		try
		{
			bmpDataInput = inputBitmap->LockBits(rect, Imaging::ImageLockMode::ReadOnly, inputBitmap->PixelFormat);
			bmpDataOutput = outputBitmap->LockBits(rect, Imaging::ImageLockMode::WriteOnly, outputBitmap->PixelFormat);
			unsigned char* ptrInput = (unsigned char*)bmpDataInput->Scan0.ToPointer();
			unsigned char* ptrOutput = (unsigned char*)bmpDataOutput->Scan0.ToPointer();
			int width = inputBitmap->Width;
			int height = inputBitmap->Height;
			int stride = bmpDataInput->Stride;

			run_inversion_filter_cuda(ptrOutput, ptrInput, width, height, stride);
		}
		finally
		{
			if (bmpDataInput != nullptr) inputBitmap->UnlockBits(bmpDataInput);
			if (bmpDataOutput != nullptr) outputBitmap->UnlockBits(bmpDataOutput);
		}

		auto end = high_resolution_clock::now();
		auto duration = duration_cast<milliseconds>(end - start);
		lb_timer->Text = "Inverter (CUDA):";
		textB_Time->Text = duration.count().ToString() + " ms";

		pbox_output->Image = outputBitmap;
		delete inputBitmap;
	}
	};
}
