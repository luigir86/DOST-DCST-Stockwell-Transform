%
% Code by: U. Battisti and L. Riba
% 7th November 2015
%
%         ST_tutorial to be used in conjunction with the ST class.
%
% This file describes the use of the DOST and DCST for 1 and 2 dimensional signals.
% It implements the fast (O(N log N)) algorithm to compute the dost introduced
% in [7] and [8] and described mathematically in [1].
%
% The script computes the DOST and the DCST of a 1 dimensional and a 2 dimensional 
% signal.
%
% A graphical explanation of the code can be found in the included ST_algorithms.pdf
%
%                                 HOW TO CITE
% If this code helps you in your research, please cite
% [1]   U. Battisti, L. Riba, "Window-dependent bases for efficient
%       representations of the Stockwell transform",
%       Applied and Computational Harmonic Analysis, 23 February 2015,
%       http://dx.doi.org/10.1016/j.acha.2015.02.002.
%
%                                 CONTACTS
% If you find a bug or you need more information regarding the code feel free
% to contact Luigi Riba at: ribaluigi (at) gmail (dot) com
%
%
%              THE S-TRANSFORM, A BRIEF INTRODUCTION
% The Stockwell transform (S-transform), introduced by R. G. Stockwell in 1996 [6],
% can be seen as an hybrid between the wavelet and Gabor transform
% (short time Fourier transform). The link between these transforms
% becomes evident in the multidimensional case as shown in [2] and [3].
% In practice, we can think of the S-transform as a short time Fourier transform
% in which the analyzing window gives better time localization for high frequencies
% and better time localization for low frequencies.
%
% The S-transform is computationally expensive. In [5], R. G. Stockwell himself 
% showed how to decompose a signal on a basis, called "DOST basis", which is linked 
% to the S-transform.
% 
% Wang and Orchard found a FFT-fast algorithm to compute the DOST coefficients, 
% see [7] and [8]. This made feasible the use of the DOST for 2 dimensional signals.
%
% The mathematical description of the fast DOST algorithmhs is based on the Plancharel 
% formula (see [1], Proposition 13 and Remark 4).
%
% In this code we also implemented the Discrete Cosine Stockwell Transform (DCST),
% which follows the DOST construction but uses the DCT instead of the FFT.
%
%                                REFERENCES
% [1]   U. Battisti, L. Riba, "Window-dependent bases for efficient
%       representations of the Stockwell transform",
%       Applied and Computational Harmonic Analysis, 23 February 2015,
%       http://dx.doi.org/10.1016/j.acha.2015.02.002.
% [2]   L. Riba, M. Wong, Continuous inversion formulas for multi-dimensional
%       modified Stockwell transforms, Integral Transforms Spec. Funct. 2015
% [3]   L. Riba, Multi-dimensional Stockwell transforms and applications,
%       PhD thesis, Università degli Studi di Torino, Italy, 2014.
% [4]   R.G. Stockwell, "Why use the S-Transform", Pseudo-differential
%       operators partial differential equations and time-frequency
%       analysis, vol. 52 Fields Inst. Commun., pages 279--309,
%       Amer. Math. Soc., Providence, RI 2007;
% [5]   R.G. Stockwell, "A basis for efficient representation of the
%       S-transform", Digital Signal Processing, 17: 371--393, 2007;
% [6]   R.G. Stockwell, L. Mansinha, R.P. Lowe, Localization of the complex spectrum:
%       the S transform, IEEE Trans. Signal Process. 44 (1996) 998–1001.
% [7]   Y. Wang and J. Orchard, "Fast-discrete orthonormal
%       Stockwell transform", SISC: 31:4000--4012, 2009;
% [8]   Y. Wang, "Efficient Stockwell transform with applications to
%       image processing", PhD thesis, University of Waterloo,
%       Ontario Canada, 2011;
%
%                             FUNCTION DESCRIPTIONS
%
%   ST.dost()
%     computes the DOST coefficients of a given signal
%     input: vector (or matrix) of size 2^k;
%     output: the DOST coefficients of the vector (of the columns of the matrix)
%
%   ST.rearrangeDost()
%     rearranges the DOST coefficients to make them more readable
%     input: a vector of DOST coefficients (length n=2^k)
%     output: a n x n matrix which describes the time frequency plane using the
%     DOST coefficients. See ST_algorithms.pdf for a graphical explanation.
%
%   ST.idost()
%     computes the inverse DOST
%     input: vector (or matrix) of size 2^k
%     output: the iDOST coefficients of the vector (of the columns of the matrix)
%
%   ST.dost2()
%     computes the 2 dimensional dost of a given matrix (ex: image) taking 
%	  the dost on the columns and then on the rows
%     input: matrix of size 2^k
%     output: dost2 coefficients of the matrix
%
%   ST.idost2()
%     computes the inverse 2 dimensional dost of a given matrix (ex: image)
%     input: matrix of size 2^k
%     output: idost2 coefficients of the matrix
%
%   ST.dcst()
%     computes the dcst coefficients of a given signal
%     input: vector (or matrix) of size 2^k
%     output: the dcst coefficients of the vector (of the columns of the matrix)
%
%   ST.rearrangeDcst()
%     rearranges the dcst coefficients to make them more readable
%     input: a vector of dost coefficients (length n=2^k)
%     output: a n x n matrix which describes the time frequency plane using the
%     dcst coefficients. See ST_algorithms.pdf for a graphical explanation
%
%   ST.idcst()
%     computes the inverse DCST, i.e. given a vector of DCST coefficients gives
%     back the original signal
%     input: vector (or matrix) of size 2^k
%     output: the idcst coefficients of the vector (of the columns of the matrix)
%
%   ST.dcst2()
%     computes the 2 dimensional dcst of a given matrix (ex: image)
%     taking the dcst on the columns and then on the rows
%     input: matrix of size 2^k
%     output: dcst2 coefficients of the matrix
%
%   ST.idcst2()
%     computes the inverse 2 dimensional dcst of a given matrix (ex: image)
%     input: matrix of size 2^k
%     output: dcst2 coefficients of the matrix
%
%   ST.fourier(), ST.ifourier(), ST.fourier2(), ST.ifourier2()
%     these functions compute the normalized and centered FFT the 1 and 2
%     dimensional cases and their inverses
%
%                             PRIVATE FUNCTIONS
%   ST.dostbw() and ST.dcstbw()
%     give the frequency bandwidth decomposition of the dost and dcst resepectively
%     input: number of samples (a power of 2)
%     output: the bandwidth decomposition.
%     examples: ST.dostbw(16) = [1 4 2 1 1 1 2 4], ST.dcstbw(16) = [1 1 2 4 8]
%
% Additional details:
% Copyright (c) by U. Battisti and L. Riba
% $Revision: 1.0 $
% $Date: 7th November 2015$

clear all
close all

% OCTAVE USERS
% if you are using Octave, install "signal" and "image" packages and load them
% before running this code (tested with Octave 4.0 on Windows 7)
%
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% 1-dimensional case
% we create a test signal and we compute its fft, dct, dost and dcst

ns = 2^10; % number of samples
t = linspace(0,1,ns); % signal of 1s (0 to 1) sampled ns-times
f = linspace(-ns/2,ns/2 -1,ns); % frequencies
in = exp(2 *pi* 1i*(ns/2^2)*t).*((0/8<=t)&(t<1/8))+...
     exp(2 *pi* 1i*(ns/2^3)*t).*((1/8<=t)&(t<2/8))+ ...
     exp(2 *pi* 1i*(ns/2^4)*t).*((2/8<=t)&(t<3/8))+...
     exp(2 *pi* 1i*(ns/2^5)*t).*((3/8<=t)&(t<4/8))+...
     exp(2 *pi* 1i*(ns/2^6)*t).*((4/8<=t)&(t<5/8))+...
     exp(2 *pi* 1i*(-ns/2^2)*t).*((5/8<=t)&(t<6/8))+ ...
     exp(2 *pi* 1i*(-ns/2^3)*t).*((6/8<=t)&(t<7/8))+...
     exp(2 *pi* 1i*(-ns/2^4)*t).*((7/8<=t)&(t<8/8));
     
if (sum(mod(log2(size(in)), 1)) == 0)
else
    error('ERROR: STransfomrs works with signal with a 2^k size.')
end

fftIn = ST.fourier(in);
dostIn = ST.dost(in);
recdostIn = ST.idost(ST.dost(in));

% to simplify the interpretation of the dost coefficients we rearrange them
% in the phase space.
readableDostIn = ST.rearrangeDost(dostIn);

figure
subplot(3, 1, 1)
plot(t, real(in))
title('input signal, real part')
xlabel('time')
ylabel('amplitude')
axis tight
subplot(3, 1, 2)
plot(f,abs(fftIn))
title('normalized and centered fft (abs)')
xlabel('frequency')
ylabel('magnitude')
axis tight
subplot(3, 1, 3)
imagesc(t,f,abs(readableDostIn))
title('dost representation')
xlabel('time')
ylabel('frequency')
axis tight

in = real(in);  % real signal
f = linspace(0,ns/2 -1,ns); % real frequencies

dctIn = dct(in);
dcstIn = ST.dcst(in);
recdcstIn = ST.idcst(ST.dcst(in));
% to simplify the interpratation of the dcst coefficients we rearrange them
% in the phase space.
readableDcstIn = ST.rearrangeDcst(dcstIn);

figure
subplot(3, 1, 1)
plot(t, in)
title('input signal')
xlabel('time')
ylabel('magnitude')
axis tight
subplot(3, 1, 2)
plot(f,dctIn)
title('dct')
xlabel('frequency')
ylabel('magnitude')
axis tight
subplot(3, 1, 3)
imagesc(t,f,readableDcstIn)
title('dcst')
xlabel('time')
ylabel('frequency')
axis tight
%
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% 2-dimensional case
% we load an image (lena512.bmp http://www.ece.rice.edu/~wakin/images/)
% and we compute its fft2, dct2, dost2, dcst2
% we show the results in a logaritmic scale to make them more readable.
% Notice that, as the dct2, the dcst encodes a large part of the image information
% in few low frequency coefficients, thus could be used for image compression.

% In case you use Matlab it is instructive to analyze the TARTAN image in order
% to get a better understanding on what are the frequency bands represented
% by the different rectangles. You can do this uncommenting the following lines
load tartan
in2d = X;

% OCTAVE users: uncomment the following line to load the standard lena image and
% comment the preceding two lines
% in2d = double(imread('lena512.bmp'));

if (sum(mod(log2(size(in2d)), 1)) == 0)
else
    error('ERROR: STransfomrs works with signal with a 2^k size.')
end

fft2In2d = ST.fourier2(in2d);
dost2In2d = ST.dost2(in2d);
recdostIn2d = ST.idost2(ST.dost2(in2d));

figure
subplot(1, 3, 1)
imagesc(in2d)
title('input signal')
xlabel('x_1')
ylabel('x_2')
axis equal tight
set(gca, 'XTickLabel', '', 'YTickLabel', '')
subplot(1, 3, 2)
imagesc(log(abs(fft2In2d) + 1))
title('normalized and centered fft2')
xlabel('\xi_1')
ylabel('\xi_2')
set(gca, 'XTickLabel', '', 'YTickLabel', '')
axis equal tight
subplot(1, 3, 3)
imagesc(log(abs(dost2In2d) + 1))
title('dost2 ')
xlabel('\xi_1')
ylabel('\xi_2')
set(gca, 'XTickLabel', '', 'YTickLabel', '')
axis equal tight
colormap gray

dct2In2d = dct2(in2d);
dcst2In2d = ST.dcst2(in2d);
recdcstIn2d = ST.idcst2(ST.dcst2(in2d));

figure
subplot(1, 3, 1)
imagesc(in2d)
title('input signal')
xlabel('x_1')
ylabel('x_2')
set(gca, 'XTickLabel', '', 'YTickLabel', '')
axis equal tight
subplot(1, 3, 2)
imagesc(log(abs(dct2In2d + 1)))
title('dct2')
xlabel('\xi_1')
ylabel('\xi_2')
set(gca, 'XTickLabel', '', 'YTickLabel', '')
axis equal tight
subplot(1, 3, 3)
imagesc(log(abs(dcst2In2d + 1)))
title('dcst2')
xlabel('\xi_1')
ylabel('\xi_2')
set(gca, 'XTickLabel', '', 'YTickLabel', '')
axis equal tight
colormap gray

