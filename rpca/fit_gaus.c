#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "nrutil.h"


void burst_gaus(float **y,int nx,int ny,float **yt,float **residu,double *para)
{
  void gasfit_2D(double **x, double *y,int np,double *para,double bg0, int fbg);
  double **xs,*ys,ymax;
  int i,j,np,npt,im,jm,**xi,k,ix,iy;
  double kc,bgc,sigmac,xc,yc,sigma2v,det;
  double bg0=0;
  double a,b,c,d,e,f,g,h;
  a=y[0][0];b=y[0][ny-1];c=y[nx-1][0];d=y[nx-1][ny-1];
  int fbg=0;
  np=nx*ny;
  xs=dmatrix(0,np-1,0,1);
  ys=dvector(0,np-1);
  xi=imatrix(0,np-1,0,1);
  a=(a<b)?a:b;c=(c<d)?c:d;ymax=(a<c)?a:c;
  //printf("ymax=%f\n",ymax,y[0][0] );
  for(i=-2;i<=2;i++)for(j=-2;j<=2;j++){
    ix=i+nx/2;iy=j+ny/2;
      if(ymax<y[ix][iy]){
        ymax=y[ix][iy];//printf("ymax=%f\n",ymax,y[0][0] );
        im=ix;jm=iy;
      }
    }//printf("im=%d,jm=%d\n",im,jm );
  npt=0;
  for(i=-2;i<=2;i++){
    for(j=-2;j<=2;j++){
      xs[npt][0]=xi[npt][0]=i+im;
      xs[npt][1]=xi[npt][1]=j+jm;
      ys[npt]=y[im+i][jm+j];
      npt++;
    }
  }
  gasfit_2D(xs, ys,npt,para,bg0,fbg);
  kc=para[0];sigmac=para[1];bgc=para[2];xc=para[3];yc=para[4];
  //printf("%e %e %e %e %e\n",kc,sigmac,bgc,xc,yc);
  sigma2v=-1./(2.*sigmac*sigmac);
  for(i=0;i<nx;i++){
    for(j=0;j<ny;j++){
    det=DSQR(i-xc)+DSQR(j-yc);
    yt[i][j]=kc*exp(det*sigma2v)+bgc;
    residu[i][j]=y[i][j]-yt[i][j];
    }
  }
  free_dmatrix(xs,0,np-1,0,1);
  free_imatrix(xi,0,np-1,0,1);
  free_dvector(ys,0,np-1);
}

void star_gaus(float **y,int nx, int ny, float **yt,float **residu,double *para)
{
  void gasfit_2D(double **x, double *y,int np,double *para,double bg0, int fbg);
  double **xs,*ys,ymax;
  int i,j,np,npt,im,jm,**xi,k,ix,iy;
  double kc,bgc,sigmac,xc,yc,sigma2v,det;
  double bg0=0;
  double a,b,c,d,e,f,g,h;
  a=y[0][0];b=y[0][ny/2-1];c=y[nx/2-1][0];d=y[nx/2-1][ny/2-1];
  int fbg=0;
  np=nx*ny;
  xs=dmatrix(0,np-1,0,1);
  ys=dvector(0,np-1);
  xi=imatrix(0,np-1,0,1);
  a=(a<b)?a:b;c=(c<d)?c:d;ymax=(a<c)?a:c;
  for(i=-5;i<=5;i++)for(j=-5;j<=5;j++){
    ix=i+nx/2;iy=j+ny/2;
      if(ymax<y[ix][iy]){ymax=y[ix][iy];im=ix;jm=iy;}
    }
  //printf("peak:%f\tmaxpos:%d,%d\n",ymax,im,jm);
  npt=0;
  for(i=-3;i<=3;i++){
    for(j=-3;j<=3;j++){
      xs[npt][0]=xi[npt][0]=i+im;
      xs[npt][1]=xi[npt][1]=j+jm;
      ys[npt]=y[im+i][jm+j];
      npt++;
    }
  }
  gasfit_2D(xs, ys,npt,para,bg0,fbg);
  //kc=para[0];sigmac=para[1];bgc=para[2];xc=para[3];yc=para[4];
  //printf("%e %e %e %e %e\n",kc,sigmac,bgc,xc,yc);
  //sigma2v=-1./(2.*sigmac*sigmac);
  //for(i=0;i<nx;i++){
  //  for(j=0;j<ny;j++){
  //  det=DSQR(i-xc)+DSQR(j-yc);
  //  yt[i][j]=kc*exp(det*sigma2v)+bgc;
  //  residu[i][j]=y[i][j]-yt[i][j];
  //  }
  //}
  free_dmatrix(xs,0,np-1,0,1);
  free_imatrix(xi,0,np-1,0,1);
  free_dvector(ys,0,np-1);
}

void galaxy_gaus(float **y,int nx, int ny,float **yt,float **residu,double *para)
{
  void gasfit_2D(double **x, double *y,int np,double *para,double bg0, int fbg);
  double **xs,*ys,ymax;
  int i,j,np,npt,im,jm,**xi,k,ix,iy;
  double kc,bgc,sigmac,xc,yc,sigma2v,det;
  double bg0=0;
  double a,b,c,d,e,f,g,h;
  a=y[0][0];b=y[0][ny/2-1];c=y[nx/2-1][0];d=y[nx/2-1][ny/2-1];
  int fbg=0;
  np=nx*ny;
  xs=dmatrix(0,np-1,0,1);
  ys=dvector(0,np-1);
  xi=imatrix(0,np-1,0,1);
  a=(a<b)?a:b;c=(c<d)?c:d;ymax=(a<c)?a:c;
  //printf("ymax=%f\n",ymax,y[0][0] );
  for(i=0;i<nx/2;i++)for(j=0;j<ny/2;j++){
      ix=i+nx/4;iy=j+ny/4;//printf("x=%d,y=%d\n",ix,iy );
      if(ymax<y[ix][iy]){
        ymax=y[ix][iy];//printf("ymax=%f\n",ymax,y[0][0] );
        im=ix;jm=iy;
      }
    }//printf("im=%d,jm=%d\n",im,jm );
  npt=0;
  for(i=-5;i<=5;i++){
    for(j=-5;j<=5;j++){
      xs[npt][0]=xi[npt][0]=i+im;
      xs[npt][1]=xi[npt][1]=j+jm;
      ys[npt]=y[im+i][jm+j];
      npt++;
    }
  }
  gasfit_2D(xs, ys,npt,para,bg0,fbg);
  //kc=para[0];sigmac=para[1];bgc=para[2];xc=para[3];yc=para[4];
  //printf("%e %e %e %e %e\n",kc,sigmac,bgc,xc,yc);
  //sigma2v=-1./(2.*sigmac*sigmac);
  //for(i=0;i<nx;i++){
  //  for(j=0;j<ny;j++){
  //  det=DSQR(i-xc)+DSQR(j-yc);
  //  yt[i][j]=kc*exp(det*sigma2v)+bgc;
  //  residu[i][j]=y[i][j]-yt[i][j];
  //  }
  //}
  free_dmatrix(xs,0,np-1,0,1);
  free_imatrix(xi,0,np-1,0,1);
  free_dvector(ys,0,np-1);
}


void gasfit_2D(double **x, double *y,int np,double *para,double bg0,int fbg)
{
  void search_2D(double **x,double *y,int np,double kc,double kd,double sigmac,
              double sigmad,double bgc,double bgd,double xc, double xd, 
     double yc, double yd,double *para,double bg0,int fbg);
  int i,j,k,imax=0,isigma;
  double ymax,ymin,kc,kd,sigmac,sigmad,bgc,bgd,ysigma,xc,yc,xd,yd;
  double det,dett;
  ymin=ymax=y[imax];
  for(i=1;i<np;i++){
    if(ymax<y[i]){imax=i;ymax=y[i];}
    if(ymin>y[i])ymin=y[i];
    
  }
  //printf("ymax=%e,ymin=%e\n",ymax,ymin);
  kc=ymax;kd=ymax/12.;
  det=ysigma=kc*exp(-0.5);
  for(i=0;i<np;i++){
    dett=fabs(ysigma-y[i]);
    if(dett<det){det=dett;isigma=i;}
  }
  xc=x[imax][0];yc=x[imax][1];
  //sigmac=sqrt(DSQR(xc-x[isigma][0])+DSQR(yc-x[isigma][1]));
  //xd=yd=sigmac*0.25;
  ////////////////////////////////////
  double dx = x[isigma][0] - xc;
  double dy = x[isigma][1] - yc;
  sigmac = sqrt(dx*dx + dy*dy); // 正确计算欧氏距离
  sigmac = fmax(sigmac, 1.0); 
  xd = yd = sigmac * 0.1;  
  ////////////////////////////////////
  sigmad=0.25*sigmac;
  bgc=0.;bgd=fabs(ymin);
  
  for(i=0;i<10;i++){
      
    //printf("%e %e %e %e %e k2=%e\n",kc,sigmac,bgc,xc,yc,para[5]);
    search_2D(x,y,np,kc,kd,sigmac,sigmad,bgc,bgd,xc,xd,yc,yd,para,bg0,fbg);
    kd*=0.33;sigmad*=0.33;bgd*=0.33;xd*=0.33;yd*=0.33;
    //kd*=0.5;sigmad*=0.5;bgd*=0.5;xd*=0.5;yd*=0.5;
    kc=para[0];sigmac=para[1];bgc=para[2];xc=para[3];yc=para[4]; 
    //printf("x%f,y=%f\t",xc,yc);
    
  }//printf("\n");
  if(fbg==0)para[2]=bg0;
}
void search_2D(double **x,double *y,int np,double kc,double kd,double sigmac,
         double sigmad,double bgc,double bgd,double xc, double xd, 
         double yc, double yd,double *para,double bg0,int fbg)
{
  double k,sigma,bg,k2,k20,sigma2v,det,xt,yt;
  int i,j,l,m,p,q;
  sigma2v=-1./(2.*sigmac*sigmac);
  bg=bgc;
  k20=0;
  for(m=0;m<np;m++){
    det=DSQR(x[m][0]-xc)+DSQR(x[m][1]-yc);
    det=kc*exp(det*sigma2v)+bgc-y[m];
    k20+=det*det;
  }             
  for(i=-4;i<=4;i++){
    k=kc+i*kd;
    if(k>0){
      for(j=-4;j<=4;j++){
        sigma=sigmac+j*sigmad;
          if(sigma>0){
            sigma2v=-1./(2.*sigma*sigma);
            for(p=-4;p<=4;p++){
              xt=xc+p*xd;
              for(q=-4;q<=4;q++){
                yt=yc+q*yd;
                k2=0;
                //double center_penalty = DSQR(xt - xc) + DSQR(yt - yc);
                //k2 += center_penalty * 0.01; // 轻微偏向原始中心
                if(fbg==0){
                  bg=bg0;
                  for(m=0;m<np;m++){
                    det=DSQR(x[m][0]-xt)+DSQR(x[m][1]-yt);
                    det=k*exp(det*sigma2v)+bg-y[m];
                    k2+=det*det;
                  }
                }
                else{
                  for(l=-4;l<=4;l++){
                    bg=bgc+l*bgd;
                    for(m=0;m<np;m++){
                      det=DSQR(x[m][0]-xt)+DSQR(x[m][1]-yt);
                      det=k*exp(det*sigma2v)+bg-y[m];
                      k2+=det*det;
                    }
                  }
                }
              //printf("k20=%e k2=%e\n",k20,k2);
              if(k2<=k20){k20=k2;para[5]=k2;
              para[0]=k;para[1]=sigma;para[2]=bg;para[3]=xt;para[4]=yt;}
            }
          }
        }   
      }
    }
  }
}



#define MAX_ITER 100
#define TOLERANCE 0.001



void galaxy_gaus1(float **y,int nx,int ny,float **yt,float **residu,double *center) {
    double find_half_light_radius(float **y, int nx, int ny, double x_init, double y_init);
                
    // 初始估计中心为图像中心
    double x_win = (nx - 1) / 2.0;
    double y_win = (ny - 1) / 2.0;

    // 使用半分法查找d50
    double d50 = find_half_light_radius(y, nx, ny, x_win, y_win);
    double s_win = d50 / sqrt(8 * log(2));

    // 迭代更新质心坐标
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        double sum_x = 0, sum_y = 0, sum_w = 0;
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                double dx = i - x_win;
                double dy = j - y_win;
                double r = sqrt(dx*dx + dy*dy);
                double w = exp(-r*r / (2*s_win*s_win));
                sum_x += w * y[i][j] * dx;
                sum_y += w * y[i][j] * dy;
                sum_w += w * y[i][j];
            }
        }
        d50 = find_half_light_radius(y, nx, ny, x_win, y_win);
        s_win = d50 / sqrt(8 * log(2));

        if (sum_w == 0) break; // 防止除以0

        double new_x_win = x_win + sum_x / sum_w;
        double new_y_win = y_win + sum_y / sum_w;

        // 检查是否达到精度要求
        if (fabs(new_x_win - x_win) < TOLERANCE && fabs(new_y_win - y_win) < TOLERANCE) {
            center[3] = new_x_win;
            center[4] = new_y_win;
            return;
        }

        x_win = new_x_win;
        y_win = new_y_win;
    }

    // 如果达到最大迭代次数仍未满足精度要求，则使用最后一次的结果
    center[3] = x_win;
    center[4] = y_win;
}

double find_half_light_radius(float **y, int nx, int ny, double x_init, double y_init) {
    double total_flux = 0;
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            total_flux += y[i][j];
        }
    }
    double half_light_flux = total_flux / 2;

    double low = 0, high = sqrt(nx*nx + ny*ny), mid;
    while ((high - low) > 0.01) { // 精度设置为0.01像素
        mid = (low + high) / 2;
        double enclosed_flux = 0;
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                double dx = i - x_init;
                double dy = j - y_init;
                if (sqrt(dx*dx + dy*dy) <= mid) {
                    enclosed_flux += y[i][j];
                }
            }
        }
        if (enclosed_flux < half_light_flux) {
            low = mid;
        } else {
            high = mid;
        }
    }
    return (low + high) / 2;
}


void galaxy_gaus2(float **y, int nx, int ny, float **yt, float **residu, double *center) {
    const double MOMENTUM = 0.3;
    const double MIN_SIGMA = 0.5;
    
    // 改进的初始估计
    double x_win = (nx - 1) / 2.0;
    double y_win = (ny - 1) / 2.0;
    
    // 先用简单质心法获得更好的初始估计
    double init_sum_x = 0, init_sum_y = 0, init_sum = 0;
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            if (y[i][j] > 0) {
                init_sum_x += y[i][j] * i;
                init_sum_y += y[i][j] * j;
                init_sum += y[i][j];
            }
        }
    }
    if (init_sum > 0) {
        x_win = init_sum_x / init_sum;
        y_win = init_sum_y / init_sum;
    }
    
    double d50 = find_half_light_radius(y, nx, ny, x_win, y_win);
    double s_win = fmax(d50 / sqrt(8 * log(2)), MIN_SIGMA);
    
    double dx_prev = 0, dy_prev = 0;
    
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        double sum_x = 0, sum_y = 0, sum_w = 0;
        
        // 每2次迭代更新一次半光半径
        if (iter % 2 == 0) {
            d50 = find_half_light_radius(y, nx, ny, x_win, y_win);
            s_win = fmax(d50 / sqrt(8 * log(2)), MIN_SIGMA);
        }
        
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                if (y[i][j] <= 0) continue;
                
                double dx = i - x_win;
                double dy = j - y_win;
                double r_sq = dx*dx + dy*dy;
                
                // 限制计算范围，提高效率
                if (r_sq > 25 * s_win * s_win) continue;
                
                double w = exp(-0.5 * r_sq / (s_win*s_win + 1e-6));
                sum_x += w * y[i][j] * dx;
                sum_y += w * y[i][j] * dy;
                sum_w += w * y[i][j];
            }
        }
        
        if (fabs(sum_w) < 1e-10) break;
        
        double dx_new = sum_x / sum_w;
        double dy_new = sum_y / sum_w;
        
        // 带动量的更新
        double dx_step = (1 - MOMENTUM) * dx_new + MOMENTUM * dx_prev;
        double dy_step = (1 - MOMENTUM) * dy_new + MOMENTUM * dy_prev;
        
        double new_x_win = x_win + dx_step;
        double new_y_win = y_win + dy_step;
        
        // 检查边界
        new_x_win = fmax(0, fmin(nx-1, new_x_win));
        new_y_win = fmax(0, fmin(ny-1, new_y_win));
        
        // 相对收敛检查
        double rel_change = sqrt(dx_step*dx_step + dy_step*dy_step) / (s_win + 1e-6);
        if (rel_change < TOLERANCE) {
            center[3] = new_x_win;
            center[4] = new_y_win;
            return;
        }
        
        x_win = new_x_win;
        y_win = new_y_win;
        dx_prev = dx_step;
        dy_prev = dy_step;
    }
    
    center[3] = x_win;
    center[4] = y_win;
}



void centriod_psflib(float *image0,int nx,int ny,double *cent){
    void star_gaus(float **y,int nx,int ny,float **yt,float **residu,double *para);
    int i,j,k;
    float **image,**starf,**residu;
    double *para;
    starf=(float **)calloc(nx,sizeof(float *));
    residu=(float **)calloc(nx,sizeof(float *));
    image=(float **)calloc(nx,sizeof(float *));
    para=(double *)calloc(10,sizeof(double));
    for(i=0;i<nx;i++){
        image[i]=(float *)calloc(ny,sizeof(float));
        starf[i]=(float *)calloc(ny,sizeof(float));
        residu[i]=(float *)calloc(ny,sizeof(float));
        for(j=0;j<ny;j++){
            k=i*ny+j;
            image[i][j]=image0[k];
        }
    }
    star_gaus(image,nx,ny,starf,residu,para);
    cent[0]=para[3];cent[1]=para[4];cent[2]=para[1];
    for(i=0;i<nx;i++){
        free(image[i]);
        free(starf[i]);
        free(residu[i]);
    }free(image);free(starf);free(residu);
    free(para);
}








