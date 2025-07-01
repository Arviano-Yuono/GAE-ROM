Target: Apply the model to NACA0012 with varying angle of attack and Reynold's number as the parameter.

* [X] Make a filter mdoule for the point cloud around the airfoil
* [ ] fix batch size issue
* [X] ensure the model can do solo encode or decode and mapping
* [ ] add plotting for solution
* [X] implement surface loss function
* [X] try Airfrans

Bimbingan pak Pram:

* [X] lanjuttin untuk prediksi velocity profile
* [ ] buat untuk ngitung Cl and Cd

TODO TODAY:

Airfrans ada masalah, datasetnya itu variasinya ada di NACA series yang dipake + AoA + Uinf. So we are going back to our original SU2 datset to find the

* [X] Find a way to get the surface mask for RANS dataset
* [X] Study and decide about the possibility of calculating Cl and Cd
* [X] Generate the whole dataset for the RANS NACA0012
* [X] Train initial model

NEXT STEP:

1. RANS

Karena jelek hasilnya, coba lagi dengan MSE loss biasa kemudian bandingnkan dengan skala2 lambda_surface

* [X] Benerin untuk dimensi_pde = 2
* [X] Benerin MSELoss sampe bagus
* [ ] Bandingkan dengan skala2 lambda_surface

2. AirFRans

Beda kasus, masalahnya semua data airfoil beda bentuk/series, ga ada yang sama. Jadi gimana caranya masukkin informasi jenisnya untuk bantu prediksi model (or maybe buat modelnya pelajari semua biar dia nebaknya secara general aka mesh agnostic).

* [X] Siapin datasetnya jadi .h5
  * [X] Inspect kesamaan jumlah nodes
  * [X] Clipping
  * [X] Edge index
* [X] Train initial
  * [X] Train pake MSELoss
  * [X] Train pake surface + vol loss

TODO TODAY

* [ ] train surface with Uy and P
* [ ] CL and CD
* [ ] Separation point predicted vs truth

Tampilkan hasil kasus paper untuk verifikasi

Error surface

plotting warnanya diperbaiki

error untuk the whole dataset

Pressure penting,  untuk analisa Cp (penting)

liat efek dari alpha -> Pressure vs Re --> Pressure

Model untuk prediksi Cf  (mungkin Cp juga) untuk mesh airfoilnya doang

TODO TOMORROW:

* [X] Implement Cp calculation

  * [X] Cp straingt from pressure ????
  * [X] surface_mask
  * [X] surface points
  * [X] norm (ternyata ga perlu)
* [ ] Make a new dataset with more detailed airfoil
* [ ] INCLUDE PARAMS TO THE DATASET
* [ ] Implement test function or method for the val_dataset

  * [ ] Calculate overall error
  * [ ] Calculate the correlation between the Pressure `<->` alpha and Pressure `<->` Re
* [ ] Try Cf prediction for the airfoil mesh (input: airfoil_mesh, Re, Alpha)
* [ ] Implement better plotting and saving

  * [ ] make a plotter using pyvista instead of matplotlib
