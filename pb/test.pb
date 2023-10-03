EnableExplicit

Enumeration ScalarType
  #ScalarType_Byte = 0
  #ScalarType_Char = 1
  #ScalarType_Short = 2
  #ScalarType_Int = 3
  #ScalarType_Long = 4
  #ScalarType_Half = 5
  #ScalarType_Float = 6
  #ScalarType_Double = 7
  #ScalarType_ComplexHalf = 8
  #ScalarType_ComplexFloat = 9
  #ScalarType_ComplexDouble = 10
  #ScalarType_Bool = 11
  #ScalarType_QInt8 = 12
  #ScalarType_QUInt8 = 13
  #ScalarType_QInt32 = 14
  #ScalarType_BFloat16 = 15
  #ScalarType_QUInt4x2 = 16
  #ScalarType_QUInt2x4 = 17
  #ScalarType_Undefined = 18
  #ScalarType_NumOptions
EndEnumeration

Import "-lstdc++" : EndImport

Import "-lpython3.11" : EndImport

Import "libc10.so" : EndImport

Import "libbackend_with_compiler.so" : EndImport

Import "libtorch_cpu.so" : EndImport

Import "libtorch_python.so" : EndImport

Import "libtorch.so" : EndImport

Import "libPBTorch.a"
  set_grad_mode(enabled.a)
  is_autograd_enabled.a()
  
  create_tensor.i(*dims.Quad, ndims.i, type.b, gradient.a)
  delete_tensor(*tensor)
  
  get_tensor_type.a(*tensor)
  get_tensor_pointer.i(*tensor)
  tensor_to_dtype.i(*tensor, type.b)
  
  load_model.i(path.p-ascii)
  delete_model(*module)
  
  forward1.i(*module, *in1)
  forward2.i(*module, *in1, *in2)
  forward3.i(*module, *in1, *in2, *in3)
EndImport


Procedure.i ImageToTensor(img.i)
  Protected Dim dimensions.q(4)
  
  dimensions(0) = 1
  dimensions(1) = 1
  dimensions(2) = ImageWidth(img)
  dimensions(3) = ImageHeight(img)
  
  Protected x.i, y.i, alpha.f, beta.f, *tensor, *tdata.Float
  
  *tensor = create_tensor(@dimensions(), 4, #ScalarType_Float, #False)
  *tdata = get_tensor_pointer(*tensor)
  
  alpha = 1.0 / 255.0
  beta = 0.0
  
  If StartDrawing(ImageOutput(img))
    For y = 0 To ImageHeight(img)-1
      For x = 0 To ImageWidth(img)-1
          *tdata\f = Red(Point(x, y)) * alpha + beta
          *tdata + SizeOf(Float)
      Next x
    Next y
    StopDrawing()
  EndIf
  
  ProcedureReturn *tensor
EndProcedure


set_grad_mode(#False)


UsePNGImageDecoder()

Define img.i

img = LoadImage(#PB_Any, "../test_five.png")

ShowLibraryViewer("Image", img)


Define *model, *tensor_in, *tensor_out

*model = load_model("test.pt")

*tensor_in = ImageToTensor(img)

*tensor_out = forward1(*model, *tensor_in)

Define i.i, *tdata.Float, digit.i, max_prob.f

*tdata = get_tensor_pointer(*tensor_out)

max_prob = -Infinity()

For i = 0 To 9
  If *tdata\f > max_prob
    digit = i
    max_prob = *tdata\f
  EndIf
  *tdata + SizeOf(Float)
Next i

Debug "Digit is probably " + digit + " with a probability of " + max_prob

delete_tensor(*tensor_out)

delete_tensor(*tensor_in)

delete_model(*model)
; IDE Options = PureBasic 6.02 LTS (Linux - x64)
; CursorPosition = 104
; FirstLine = 89
; Folding = -
; EnableXP
; Executable = test.elf