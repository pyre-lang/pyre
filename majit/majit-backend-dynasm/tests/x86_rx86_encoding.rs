//! `rx86` memory encodings checked against dynasm static-register forms.
#![cfg(target_arch = "x86_64")]

use dynasmrt::dynasm;
use dynasmrt::x64::X64Relocation;
use dynasmrt::{DynasmApi, VecAssembler};
use majit_backend_dynasm::x86::rx86;

/// Assemble `$emit` into `got` and `$($insn)*` with dynasm, then compare bytes.
macro_rules! enc {
    ($mc:ident => $emit:expr, $($insn:tt)*) => {{
        let mut $mc = VecAssembler::<X64Relocation>::new(0);
        $emit;
        let got = $mc.finalize().unwrap();
        let mut expect = VecAssembler::<X64Relocation>::new(0);
        dynasm!(expect ; .arch x64 ; $($insn)*);
        let expect = expect.finalize().unwrap();
        assert_eq!(got, expect, "got {got:02x?} expect {expect:02x?}");
    }};
}

/// Compare encoder output to exact bytes (`rex_fw` emits `0x40`).
macro_rules! enc_bytes {
    ($mc:ident => $emit:expr, $bytes:expr) => {{
        let mut $mc = VecAssembler::<X64Relocation>::new(0);
        $emit;
        let got = $mc.finalize().unwrap();
        assert_eq!(&got[..], &$bytes[..], "got {got:02x?}");
    }};
}

#[test]
fn rx86_encoding() {
    // mov_rm
    enc!(mc => rx86::mov_rm(&mut mc, 0, (0, 0)), mov rax, [rax]);
    enc!(mc => rx86::mov_rm(&mut mc, 0, (0, 8)), mov rax, [rax + 8]);
    enc!(mc => rx86::mov_rm(&mut mc, 0, (0, -129)), mov rax, [rax - 129]);
    enc!(mc => rx86::mov_rm(&mut mc, 0, (0, 0x12345)), mov rax, [rax + 0x12345]);
    enc!(mc => rx86::mov_rm(&mut mc, 0, (4, 0)), mov rax, [rsp]);
    enc!(mc => rx86::mov_rm(&mut mc, 0, (4, 8)), mov rax, [rsp + 8]);
    enc!(mc => rx86::mov_rm(&mut mc, 0, (12, 0)), mov rax, [r12]);
    enc!(mc => rx86::mov_rm(&mut mc, 0, (12, 8)), mov rax, [r12 + 8]);
    enc!(mc => rx86::mov_rm(&mut mc, 0, (5, 0)), mov rax, [rbp]);
    enc!(mc => rx86::mov_rm(&mut mc, 0, (13, 0)), mov rax, [r13]);
    enc!(mc => rx86::mov_rm(&mut mc, 15, (15, 8)), mov r15, [r15 + 8]);
    // mov_mr
    enc!(mc => rx86::mov_mr(&mut mc, (0, 0), 0), mov [rax], rax);
    enc!(mc => rx86::mov_mr(&mut mc, (0, 8), 0), mov [rax + 8], rax);
    enc!(mc => rx86::mov_mr(&mut mc, (0, -129), 0), mov [rax - 129], rax);
    enc!(mc => rx86::mov_mr(&mut mc, (0, 0x12345), 0), mov [rax + 0x12345], rax);
    enc!(mc => rx86::mov_mr(&mut mc, (4, 0), 0), mov [rsp], rax);
    enc!(mc => rx86::mov_mr(&mut mc, (4, 8), 0), mov [rsp + 8], rax);
    enc!(mc => rx86::mov_mr(&mut mc, (12, 0), 0), mov [r12], rax);
    enc!(mc => rx86::mov_mr(&mut mc, (12, 8), 0), mov [r12 + 8], rax);
    enc!(mc => rx86::mov_mr(&mut mc, (5, 0), 0), mov [rbp], rax);
    enc!(mc => rx86::mov_mr(&mut mc, (13, 0), 0), mov [r13], rax);
    enc!(mc => rx86::mov_mr(&mut mc, (15, 8), 15), mov [r15 + 8], r15);
    // mov32_rm
    enc!(mc => rx86::mov32_rm(&mut mc, 0, (0, 0)), mov eax, [rax]);
    enc!(mc => rx86::mov32_rm(&mut mc, 0, (0, 8)), mov eax, [rax + 8]);
    enc!(mc => rx86::mov32_rm(&mut mc, 0, (0, -129)), mov eax, [rax - 129]);
    enc!(mc => rx86::mov32_rm(&mut mc, 0, (0, 0x12345)), mov eax, [rax + 0x12345]);
    enc!(mc => rx86::mov32_rm(&mut mc, 0, (4, 0)), mov eax, [rsp]);
    enc!(mc => rx86::mov32_rm(&mut mc, 0, (4, 8)), mov eax, [rsp + 8]);
    enc!(mc => rx86::mov32_rm(&mut mc, 0, (12, 0)), mov eax, [r12]);
    enc!(mc => rx86::mov32_rm(&mut mc, 0, (12, 8)), mov eax, [r12 + 8]);
    enc!(mc => rx86::mov32_rm(&mut mc, 0, (5, 0)), mov eax, [rbp]);
    enc!(mc => rx86::mov32_rm(&mut mc, 0, (13, 0)), mov eax, [r13]);
    enc!(mc => rx86::mov32_rm(&mut mc, 15, (15, 8)), mov r15d, [r15 + 8]);
    // mov32_mr
    enc!(mc => rx86::mov32_mr(&mut mc, (0, 0), 0), mov [rax], eax);
    enc!(mc => rx86::mov32_mr(&mut mc, (0, 8), 0), mov [rax + 8], eax);
    enc!(mc => rx86::mov32_mr(&mut mc, (0, -129), 0), mov [rax - 129], eax);
    enc!(mc => rx86::mov32_mr(&mut mc, (0, 0x12345), 0), mov [rax + 0x12345], eax);
    enc!(mc => rx86::mov32_mr(&mut mc, (4, 0), 0), mov [rsp], eax);
    enc!(mc => rx86::mov32_mr(&mut mc, (4, 8), 0), mov [rsp + 8], eax);
    enc!(mc => rx86::mov32_mr(&mut mc, (12, 0), 0), mov [r12], eax);
    enc!(mc => rx86::mov32_mr(&mut mc, (12, 8), 0), mov [r12 + 8], eax);
    enc!(mc => rx86::mov32_mr(&mut mc, (5, 0), 0), mov [rbp], eax);
    enc!(mc => rx86::mov32_mr(&mut mc, (13, 0), 0), mov [r13], eax);
    enc!(mc => rx86::mov32_mr(&mut mc, (15, 8), 15), mov [r15 + 8], r15d);
    // mov16_mr
    enc!(mc => rx86::mov16_mr(&mut mc, (0, 0), 0), mov [rax], ax);
    enc!(mc => rx86::mov16_mr(&mut mc, (0, 8), 0), mov [rax + 8], ax);
    enc!(mc => rx86::mov16_mr(&mut mc, (0, -129), 0), mov [rax - 129], ax);
    enc!(mc => rx86::mov16_mr(&mut mc, (0, 0x12345), 0), mov [rax + 0x12345], ax);
    enc!(mc => rx86::mov16_mr(&mut mc, (4, 0), 0), mov [rsp], ax);
    enc!(mc => rx86::mov16_mr(&mut mc, (4, 8), 0), mov [rsp + 8], ax);
    enc!(mc => rx86::mov16_mr(&mut mc, (12, 0), 0), mov [r12], ax);
    enc!(mc => rx86::mov16_mr(&mut mc, (12, 8), 0), mov [r12 + 8], ax);
    enc!(mc => rx86::mov16_mr(&mut mc, (5, 0), 0), mov [rbp], ax);
    enc!(mc => rx86::mov16_mr(&mut mc, (13, 0), 0), mov [r13], ax);
    enc!(mc => rx86::mov16_mr(&mut mc, (15, 8), 15), mov [r15 + 8], r15w);
    // mov8_mr
    // `rex_fw` (`encode_rex`) always emits a REX byte, even `0x40`.
    enc_bytes!(mc => rx86::mov8_mr(&mut mc, (0, 0), 0), [0x40, 0x88, 0x00]);
    // `rex_fw` (`encode_rex`) always emits a REX byte, even `0x40`.
    enc_bytes!(mc => rx86::mov8_mr(&mut mc, (0, 8), 0), [0x40, 0x88, 0x40, 0x08]);
    // `rex_fw` (`encode_rex`) always emits a REX byte, even `0x40`.
    enc_bytes!(mc => rx86::mov8_mr(&mut mc, (0, -129), 0), [0x40, 0x88, 0x80, 0x7F, 0xFF, 0xFF, 0xFF]);
    // `rex_fw` (`encode_rex`) always emits a REX byte, even `0x40`.
    enc_bytes!(mc => rx86::mov8_mr(&mut mc, (0, 0x12345), 0), [0x40, 0x88, 0x80, 0x45, 0x23, 0x01, 0x00]);
    // `rex_fw` (`encode_rex`) always emits a REX byte, even `0x40`.
    enc_bytes!(mc => rx86::mov8_mr(&mut mc, (4, 0), 0), [0x40, 0x88, 0x04, 0x24]);
    // `rex_fw` (`encode_rex`) always emits a REX byte, even `0x40`.
    enc_bytes!(mc => rx86::mov8_mr(&mut mc, (4, 8), 0), [0x40, 0x88, 0x44, 0x24, 0x08]);
    enc!(mc => rx86::mov8_mr(&mut mc, (0, 0), 6), mov BYTE [rax], sil);
    enc!(mc => rx86::mov8_mr(&mut mc, (12, 0), 0), mov BYTE [r12], al);
    enc!(mc => rx86::mov8_mr(&mut mc, (12, 8), 0), mov BYTE [r12 + 8], al);
    // `rex_fw` (`encode_rex`) always emits a REX byte, even `0x40`.
    enc_bytes!(mc => rx86::mov8_mr(&mut mc, (5, 0), 0), [0x40, 0x88, 0x45, 0x00]);
    enc!(mc => rx86::mov8_mr(&mut mc, (13, 0), 0), mov BYTE [r13], al);
    enc!(mc => rx86::mov8_mr(&mut mc, (15, 8), 8), mov BYTE [r15 + 8], r8b);
    // mov_mi
    enc!(mc => rx86::mov_mi(&mut mc, (0, 0), -2), mov QWORD [rax], -2);
    enc!(mc => rx86::mov_mi(&mut mc, (0, 8), -2), mov QWORD [rax + 8], -2);
    enc!(mc => rx86::mov_mi(&mut mc, (0, -129), -2), mov QWORD [rax - 129], -2);
    enc!(mc => rx86::mov_mi(&mut mc, (0, 0x12345), -2), mov QWORD [rax + 0x12345], -2);
    enc!(mc => rx86::mov_mi(&mut mc, (4, 0), -2), mov QWORD [rsp], -2);
    enc!(mc => rx86::mov_mi(&mut mc, (4, 8), -2), mov QWORD [rsp + 8], -2);
    enc!(mc => rx86::mov_mi(&mut mc, (12, 0), -2), mov QWORD [r12], -2);
    enc!(mc => rx86::mov_mi(&mut mc, (12, 8), -2), mov QWORD [r12 + 8], -2);
    enc!(mc => rx86::mov_mi(&mut mc, (5, 0), -2), mov QWORD [rbp], -2);
    enc!(mc => rx86::mov_mi(&mut mc, (13, 0), -2), mov QWORD [r13], -2);
    enc!(mc => rx86::mov_mi(&mut mc, (15, 8), -2), mov QWORD [r15 + 8], -2);
    // mov32_mi
    enc!(mc => rx86::mov32_mi(&mut mc, (0, 0), 0x123456), mov DWORD [rax], 0x123456);
    enc!(mc => rx86::mov32_mi(&mut mc, (0, 8), 0x123456), mov DWORD [rax + 8], 0x123456);
    enc!(mc => rx86::mov32_mi(&mut mc, (0, -129), 0x123456), mov DWORD [rax - 129], 0x123456);
    enc!(mc => rx86::mov32_mi(&mut mc, (0, 0x12345), 0x123456), mov DWORD [rax + 0x12345], 0x123456);
    enc!(mc => rx86::mov32_mi(&mut mc, (4, 0), 0x123456), mov DWORD [rsp], 0x123456);
    enc!(mc => rx86::mov32_mi(&mut mc, (4, 8), 0x123456), mov DWORD [rsp + 8], 0x123456);
    enc!(mc => rx86::mov32_mi(&mut mc, (12, 0), 0x123456), mov DWORD [r12], 0x123456);
    enc!(mc => rx86::mov32_mi(&mut mc, (12, 8), 0x123456), mov DWORD [r12 + 8], 0x123456);
    enc!(mc => rx86::mov32_mi(&mut mc, (5, 0), 0x123456), mov DWORD [rbp], 0x123456);
    enc!(mc => rx86::mov32_mi(&mut mc, (13, 0), 0x123456), mov DWORD [r13], 0x123456);
    enc!(mc => rx86::mov32_mi(&mut mc, (15, 8), 0x123456), mov DWORD [r15 + 8], 0x123456);
    // mov16_mi
    enc!(mc => rx86::mov16_mi(&mut mc, (0, 0), 0x1234), mov WORD [rax], 0x1234);
    enc!(mc => rx86::mov16_mi(&mut mc, (0, 8), 0x1234), mov WORD [rax + 8], 0x1234);
    enc!(mc => rx86::mov16_mi(&mut mc, (0, -129), 0x1234), mov WORD [rax - 129], 0x1234);
    enc!(mc => rx86::mov16_mi(&mut mc, (0, 0x12345), 0x1234), mov WORD [rax + 0x12345], 0x1234);
    enc!(mc => rx86::mov16_mi(&mut mc, (4, 0), 0x1234), mov WORD [rsp], 0x1234);
    enc!(mc => rx86::mov16_mi(&mut mc, (4, 8), 0x1234), mov WORD [rsp + 8], 0x1234);
    enc!(mc => rx86::mov16_mi(&mut mc, (12, 0), 0x1234), mov WORD [r12], 0x1234);
    enc!(mc => rx86::mov16_mi(&mut mc, (12, 8), 0x1234), mov WORD [r12 + 8], 0x1234);
    enc!(mc => rx86::mov16_mi(&mut mc, (5, 0), 0x1234), mov WORD [rbp], 0x1234);
    enc!(mc => rx86::mov16_mi(&mut mc, (13, 0), 0x1234), mov WORD [r13], 0x1234);
    enc!(mc => rx86::mov16_mi(&mut mc, (15, 8), 0x1234), mov WORD [r15 + 8], 0x1234);
    // mov8_mi
    // `rex_fw` (`encode_rex`) always emits a REX byte, even `0x40`.
    enc_bytes!(mc => rx86::mov8_mi(&mut mc, (0, 0), -5), [0x40, 0xC6, 0x00, 0xFB]);
    // `rex_fw` (`encode_rex`) always emits a REX byte, even `0x40`.
    enc_bytes!(mc => rx86::mov8_mi(&mut mc, (0, 8), -5), [0x40, 0xC6, 0x40, 0x08, 0xFB]);
    // `rex_fw` (`encode_rex`) always emits a REX byte, even `0x40`.
    enc_bytes!(mc => rx86::mov8_mi(&mut mc, (0, -129), -5), [0x40, 0xC6, 0x80, 0x7F, 0xFF, 0xFF, 0xFF, 0xFB]);
    // `rex_fw` (`encode_rex`) always emits a REX byte, even `0x40`.
    enc_bytes!(mc => rx86::mov8_mi(&mut mc, (0, 0x12345), -5), [0x40, 0xC6, 0x80, 0x45, 0x23, 0x01, 0x00, 0xFB]);
    // `rex_fw` (`encode_rex`) always emits a REX byte, even `0x40`.
    enc_bytes!(mc => rx86::mov8_mi(&mut mc, (4, 0), -5), [0x40, 0xC6, 0x04, 0x24, 0xFB]);
    // `rex_fw` (`encode_rex`) always emits a REX byte, even `0x40`.
    enc_bytes!(mc => rx86::mov8_mi(&mut mc, (4, 8), -5), [0x40, 0xC6, 0x44, 0x24, 0x08, 0xFB]);
    enc!(mc => rx86::mov8_mi(&mut mc, (12, 0), -5), mov BYTE [r12], -5);
    enc!(mc => rx86::mov8_mi(&mut mc, (12, 8), -5), mov BYTE [r12 + 8], -5);
    // `rex_fw` (`encode_rex`) always emits a REX byte, even `0x40`.
    enc_bytes!(mc => rx86::mov8_mi(&mut mc, (5, 0), -5), [0x40, 0xC6, 0x45, 0x00, 0xFB]);
    enc!(mc => rx86::mov8_mi(&mut mc, (13, 0), -5), mov BYTE [r13], -5);
    enc!(mc => rx86::mov8_mi(&mut mc, (15, 8), -5), mov BYTE [r15 + 8], -5);
    // lea_rm
    enc!(mc => rx86::lea_rm(&mut mc, 0, (0, 0)), lea rax, [rax]);
    enc!(mc => rx86::lea_rm(&mut mc, 0, (0, 8)), lea rax, [rax + 8]);
    enc!(mc => rx86::lea_rm(&mut mc, 0, (0, -129)), lea rax, [rax - 129]);
    enc!(mc => rx86::lea_rm(&mut mc, 0, (0, 0x12345)), lea rax, [rax + 0x12345]);
    enc!(mc => rx86::lea_rm(&mut mc, 0, (4, 0)), lea rax, [rsp]);
    enc!(mc => rx86::lea_rm(&mut mc, 0, (4, 8)), lea rax, [rsp + 8]);
    enc!(mc => rx86::lea_rm(&mut mc, 0, (12, 0)), lea rax, [r12]);
    enc!(mc => rx86::lea_rm(&mut mc, 0, (12, 8)), lea rax, [r12 + 8]);
    enc!(mc => rx86::lea_rm(&mut mc, 0, (5, 0)), lea rax, [rbp]);
    enc!(mc => rx86::lea_rm(&mut mc, 0, (13, 0)), lea rax, [r13]);
    enc!(mc => rx86::lea_rm(&mut mc, 15, (15, 8)), lea r15, [r15 + 8]);
    // movzx8_rm
    enc!(mc => rx86::movzx8_rm(&mut mc, 0, (0, 0)), movzx rax, BYTE [rax]);
    enc!(mc => rx86::movzx8_rm(&mut mc, 0, (0, 8)), movzx rax, BYTE [rax + 8]);
    enc!(mc => rx86::movzx8_rm(&mut mc, 0, (0, -129)), movzx rax, BYTE [rax - 129]);
    enc!(mc => rx86::movzx8_rm(&mut mc, 0, (0, 0x12345)), movzx rax, BYTE [rax + 0x12345]);
    enc!(mc => rx86::movzx8_rm(&mut mc, 0, (4, 0)), movzx rax, BYTE [rsp]);
    enc!(mc => rx86::movzx8_rm(&mut mc, 0, (4, 8)), movzx rax, BYTE [rsp + 8]);
    enc!(mc => rx86::movzx8_rm(&mut mc, 0, (12, 0)), movzx rax, BYTE [r12]);
    enc!(mc => rx86::movzx8_rm(&mut mc, 0, (12, 8)), movzx rax, BYTE [r12 + 8]);
    enc!(mc => rx86::movzx8_rm(&mut mc, 0, (5, 0)), movzx rax, BYTE [rbp]);
    enc!(mc => rx86::movzx8_rm(&mut mc, 0, (13, 0)), movzx rax, BYTE [r13]);
    enc!(mc => rx86::movzx8_rm(&mut mc, 15, (15, 8)), movzx r15, BYTE [r15 + 8]);
    // movzx16_rm
    enc!(mc => rx86::movzx16_rm(&mut mc, 0, (0, 0)), movzx rax, WORD [rax]);
    enc!(mc => rx86::movzx16_rm(&mut mc, 0, (0, 8)), movzx rax, WORD [rax + 8]);
    enc!(mc => rx86::movzx16_rm(&mut mc, 0, (0, -129)), movzx rax, WORD [rax - 129]);
    enc!(mc => rx86::movzx16_rm(&mut mc, 0, (0, 0x12345)), movzx rax, WORD [rax + 0x12345]);
    enc!(mc => rx86::movzx16_rm(&mut mc, 0, (4, 0)), movzx rax, WORD [rsp]);
    enc!(mc => rx86::movzx16_rm(&mut mc, 0, (4, 8)), movzx rax, WORD [rsp + 8]);
    enc!(mc => rx86::movzx16_rm(&mut mc, 0, (12, 0)), movzx rax, WORD [r12]);
    enc!(mc => rx86::movzx16_rm(&mut mc, 0, (12, 8)), movzx rax, WORD [r12 + 8]);
    enc!(mc => rx86::movzx16_rm(&mut mc, 0, (5, 0)), movzx rax, WORD [rbp]);
    enc!(mc => rx86::movzx16_rm(&mut mc, 0, (13, 0)), movzx rax, WORD [r13]);
    enc!(mc => rx86::movzx16_rm(&mut mc, 15, (15, 8)), movzx r15, WORD [r15 + 8]);
    // movsx8_rm
    enc!(mc => rx86::movsx8_rm(&mut mc, 0, (0, 0)), movsx rax, BYTE [rax]);
    enc!(mc => rx86::movsx8_rm(&mut mc, 0, (0, 8)), movsx rax, BYTE [rax + 8]);
    enc!(mc => rx86::movsx8_rm(&mut mc, 0, (0, -129)), movsx rax, BYTE [rax - 129]);
    enc!(mc => rx86::movsx8_rm(&mut mc, 0, (0, 0x12345)), movsx rax, BYTE [rax + 0x12345]);
    enc!(mc => rx86::movsx8_rm(&mut mc, 0, (4, 0)), movsx rax, BYTE [rsp]);
    enc!(mc => rx86::movsx8_rm(&mut mc, 0, (4, 8)), movsx rax, BYTE [rsp + 8]);
    enc!(mc => rx86::movsx8_rm(&mut mc, 0, (12, 0)), movsx rax, BYTE [r12]);
    enc!(mc => rx86::movsx8_rm(&mut mc, 0, (12, 8)), movsx rax, BYTE [r12 + 8]);
    enc!(mc => rx86::movsx8_rm(&mut mc, 0, (5, 0)), movsx rax, BYTE [rbp]);
    enc!(mc => rx86::movsx8_rm(&mut mc, 0, (13, 0)), movsx rax, BYTE [r13]);
    enc!(mc => rx86::movsx8_rm(&mut mc, 15, (15, 8)), movsx r15, BYTE [r15 + 8]);
    // movsx16_rm
    enc!(mc => rx86::movsx16_rm(&mut mc, 0, (0, 0)), movsx rax, WORD [rax]);
    enc!(mc => rx86::movsx16_rm(&mut mc, 0, (0, 8)), movsx rax, WORD [rax + 8]);
    enc!(mc => rx86::movsx16_rm(&mut mc, 0, (0, -129)), movsx rax, WORD [rax - 129]);
    enc!(mc => rx86::movsx16_rm(&mut mc, 0, (0, 0x12345)), movsx rax, WORD [rax + 0x12345]);
    enc!(mc => rx86::movsx16_rm(&mut mc, 0, (4, 0)), movsx rax, WORD [rsp]);
    enc!(mc => rx86::movsx16_rm(&mut mc, 0, (4, 8)), movsx rax, WORD [rsp + 8]);
    enc!(mc => rx86::movsx16_rm(&mut mc, 0, (12, 0)), movsx rax, WORD [r12]);
    enc!(mc => rx86::movsx16_rm(&mut mc, 0, (12, 8)), movsx rax, WORD [r12 + 8]);
    enc!(mc => rx86::movsx16_rm(&mut mc, 0, (5, 0)), movsx rax, WORD [rbp]);
    enc!(mc => rx86::movsx16_rm(&mut mc, 0, (13, 0)), movsx rax, WORD [r13]);
    enc!(mc => rx86::movsx16_rm(&mut mc, 15, (15, 8)), movsx r15, WORD [r15 + 8]);
    // movsx32_rm
    enc!(mc => rx86::movsx32_rm(&mut mc, 0, (0, 0)), movsxd rax, DWORD [rax]);
    enc!(mc => rx86::movsx32_rm(&mut mc, 0, (0, 8)), movsxd rax, DWORD [rax + 8]);
    enc!(mc => rx86::movsx32_rm(&mut mc, 0, (0, -129)), movsxd rax, DWORD [rax - 129]);
    enc!(mc => rx86::movsx32_rm(&mut mc, 0, (0, 0x12345)), movsxd rax, DWORD [rax + 0x12345]);
    enc!(mc => rx86::movsx32_rm(&mut mc, 0, (4, 0)), movsxd rax, DWORD [rsp]);
    enc!(mc => rx86::movsx32_rm(&mut mc, 0, (4, 8)), movsxd rax, DWORD [rsp + 8]);
    enc!(mc => rx86::movsx32_rm(&mut mc, 0, (12, 0)), movsxd rax, DWORD [r12]);
    enc!(mc => rx86::movsx32_rm(&mut mc, 0, (12, 8)), movsxd rax, DWORD [r12 + 8]);
    enc!(mc => rx86::movsx32_rm(&mut mc, 0, (5, 0)), movsxd rax, DWORD [rbp]);
    enc!(mc => rx86::movsx32_rm(&mut mc, 0, (13, 0)), movsxd rax, DWORD [r13]);
    enc!(mc => rx86::movsx32_rm(&mut mc, 15, (15, 8)), movsxd r15, DWORD [r15 + 8]);
    // movsd_xm
    enc!(mc => rx86::movsd_xm(&mut mc, 0, (0, 0)), movsd xmm0, [rax]);
    enc!(mc => rx86::movsd_xm(&mut mc, 0, (0, 8)), movsd xmm0, [rax + 8]);
    enc!(mc => rx86::movsd_xm(&mut mc, 0, (0, -129)), movsd xmm0, [rax - 129]);
    enc!(mc => rx86::movsd_xm(&mut mc, 0, (0, 0x12345)), movsd xmm0, [rax + 0x12345]);
    enc!(mc => rx86::movsd_xm(&mut mc, 0, (4, 0)), movsd xmm0, [rsp]);
    enc!(mc => rx86::movsd_xm(&mut mc, 0, (4, 8)), movsd xmm0, [rsp + 8]);
    enc!(mc => rx86::movsd_xm(&mut mc, 0, (12, 0)), movsd xmm0, [r12]);
    enc!(mc => rx86::movsd_xm(&mut mc, 0, (12, 8)), movsd xmm0, [r12 + 8]);
    enc!(mc => rx86::movsd_xm(&mut mc, 0, (5, 0)), movsd xmm0, [rbp]);
    enc!(mc => rx86::movsd_xm(&mut mc, 0, (13, 0)), movsd xmm0, [r13]);
    enc!(mc => rx86::movsd_xm(&mut mc, 15, (15, 8)), movsd xmm15, [r15 + 8]);
    // movsd_mx
    enc!(mc => rx86::movsd_mx(&mut mc, (0, 0), 0), movsd [rax], xmm0);
    enc!(mc => rx86::movsd_mx(&mut mc, (0, 8), 0), movsd [rax + 8], xmm0);
    enc!(mc => rx86::movsd_mx(&mut mc, (0, -129), 0), movsd [rax - 129], xmm0);
    enc!(mc => rx86::movsd_mx(&mut mc, (0, 0x12345), 0), movsd [rax + 0x12345], xmm0);
    enc!(mc => rx86::movsd_mx(&mut mc, (4, 0), 0), movsd [rsp], xmm0);
    enc!(mc => rx86::movsd_mx(&mut mc, (4, 8), 0), movsd [rsp + 8], xmm0);
    enc!(mc => rx86::movsd_mx(&mut mc, (12, 0), 0), movsd [r12], xmm0);
    enc!(mc => rx86::movsd_mx(&mut mc, (12, 8), 0), movsd [r12 + 8], xmm0);
    enc!(mc => rx86::movsd_mx(&mut mc, (5, 0), 0), movsd [rbp], xmm0);
    enc!(mc => rx86::movsd_mx(&mut mc, (13, 0), 0), movsd [r13], xmm0);
    enc!(mc => rx86::movsd_mx(&mut mc, (15, 8), 15), movsd [r15 + 8], xmm15);
    // movss_xm
    enc!(mc => rx86::movss_xm(&mut mc, 0, (0, 0)), movss xmm0, [rax]);
    enc!(mc => rx86::movss_xm(&mut mc, 0, (0, 8)), movss xmm0, [rax + 8]);
    enc!(mc => rx86::movss_xm(&mut mc, 0, (0, -129)), movss xmm0, [rax - 129]);
    enc!(mc => rx86::movss_xm(&mut mc, 0, (0, 0x12345)), movss xmm0, [rax + 0x12345]);
    enc!(mc => rx86::movss_xm(&mut mc, 0, (4, 0)), movss xmm0, [rsp]);
    enc!(mc => rx86::movss_xm(&mut mc, 0, (4, 8)), movss xmm0, [rsp + 8]);
    enc!(mc => rx86::movss_xm(&mut mc, 0, (12, 0)), movss xmm0, [r12]);
    enc!(mc => rx86::movss_xm(&mut mc, 0, (12, 8)), movss xmm0, [r12 + 8]);
    enc!(mc => rx86::movss_xm(&mut mc, 0, (5, 0)), movss xmm0, [rbp]);
    enc!(mc => rx86::movss_xm(&mut mc, 0, (13, 0)), movss xmm0, [r13]);
    enc!(mc => rx86::movss_xm(&mut mc, 15, (15, 8)), movss xmm15, [r15 + 8]);
    // movss_mx
    enc!(mc => rx86::movss_mx(&mut mc, (0, 0), 0), movss [rax], xmm0);
    enc!(mc => rx86::movss_mx(&mut mc, (0, 8), 0), movss [rax + 8], xmm0);
    enc!(mc => rx86::movss_mx(&mut mc, (0, -129), 0), movss [rax - 129], xmm0);
    enc!(mc => rx86::movss_mx(&mut mc, (0, 0x12345), 0), movss [rax + 0x12345], xmm0);
    enc!(mc => rx86::movss_mx(&mut mc, (4, 0), 0), movss [rsp], xmm0);
    enc!(mc => rx86::movss_mx(&mut mc, (4, 8), 0), movss [rsp + 8], xmm0);
    enc!(mc => rx86::movss_mx(&mut mc, (12, 0), 0), movss [r12], xmm0);
    enc!(mc => rx86::movss_mx(&mut mc, (12, 8), 0), movss [r12 + 8], xmm0);
    enc!(mc => rx86::movss_mx(&mut mc, (5, 0), 0), movss [rbp], xmm0);
    enc!(mc => rx86::movss_mx(&mut mc, (13, 0), 0), movss [r13], xmm0);
    enc!(mc => rx86::movss_mx(&mut mc, (15, 8), 15), movss [r15 + 8], xmm15);
    // movups_mx
    enc!(mc => rx86::movups_mx(&mut mc, (0, 0), 0), movups [rax], xmm0);
    enc!(mc => rx86::movups_mx(&mut mc, (0, 8), 0), movups [rax + 8], xmm0);
    enc!(mc => rx86::movups_mx(&mut mc, (0, -129), 0), movups [rax - 129], xmm0);
    enc!(mc => rx86::movups_mx(&mut mc, (0, 0x12345), 0), movups [rax + 0x12345], xmm0);
    enc!(mc => rx86::movups_mx(&mut mc, (4, 0), 0), movups [rsp], xmm0);
    enc!(mc => rx86::movups_mx(&mut mc, (4, 8), 0), movups [rsp + 8], xmm0);
    enc!(mc => rx86::movups_mx(&mut mc, (12, 0), 0), movups [r12], xmm0);
    enc!(mc => rx86::movups_mx(&mut mc, (12, 8), 0), movups [r12 + 8], xmm0);
    enc!(mc => rx86::movups_mx(&mut mc, (5, 0), 0), movups [rbp], xmm0);
    enc!(mc => rx86::movups_mx(&mut mc, (13, 0), 0), movups [r13], xmm0);
    enc!(mc => rx86::movups_mx(&mut mc, (15, 8), 15), movups [r15 + 8], xmm15);
    // test8_mi
    enc!(mc => rx86::test8_mi(&mut mc, (0, 0), 0x3f), test BYTE [rax], 0x3f);
    enc!(mc => rx86::test8_mi(&mut mc, (0, 8), 0x3f), test BYTE [rax + 8], 0x3f);
    enc!(mc => rx86::test8_mi(&mut mc, (0, -129), 0x3f), test BYTE [rax - 129], 0x3f);
    enc!(mc => rx86::test8_mi(&mut mc, (0, 0x12345), 0x3f), test BYTE [rax + 0x12345], 0x3f);
    enc!(mc => rx86::test8_mi(&mut mc, (4, 0), 0x3f), test BYTE [rsp], 0x3f);
    enc!(mc => rx86::test8_mi(&mut mc, (4, 8), 0x3f), test BYTE [rsp + 8], 0x3f);
    enc!(mc => rx86::test8_mi(&mut mc, (12, 0), 0x3f), test BYTE [r12], 0x3f);
    enc!(mc => rx86::test8_mi(&mut mc, (12, 8), 0x3f), test BYTE [r12 + 8], 0x3f);
    enc!(mc => rx86::test8_mi(&mut mc, (5, 0), 0x3f), test BYTE [rbp], 0x3f);
    enc!(mc => rx86::test8_mi(&mut mc, (13, 0), 0x3f), test BYTE [r13], 0x3f);
    enc!(mc => rx86::test8_mi(&mut mc, (15, 8), 0x3f), test BYTE [r15 + 8], 0x3f);
    // cmp_mi
    enc!(mc => rx86::cmp_mi(&mut mc, (0, 0), 0x12345), cmp QWORD [rax], 0x12345);
    enc!(mc => rx86::cmp_mi(&mut mc, (0, 8), 0x12345), cmp QWORD [rax + 8], 0x12345);
    enc!(mc => rx86::cmp_mi(&mut mc, (0, -129), 0x12345), cmp QWORD [rax - 129], 0x12345);
    enc!(mc => rx86::cmp_mi(&mut mc, (0, 0x12345), 0x12345), cmp QWORD [rax + 0x12345], 0x12345);
    enc!(mc => rx86::cmp_mi(&mut mc, (4, 0), 0x12345), cmp QWORD [rsp], 0x12345);
    enc!(mc => rx86::cmp_mi(&mut mc, (4, 8), 0x12345), cmp QWORD [rsp + 8], 0x12345);
    enc!(mc => rx86::cmp_mi(&mut mc, (12, 0), 0x12345), cmp QWORD [r12], 0x12345);
    enc!(mc => rx86::cmp_mi(&mut mc, (12, 8), 0x12345), cmp QWORD [r12 + 8], 0x12345);
    enc!(mc => rx86::cmp_mi(&mut mc, (5, 0), 0x12345), cmp QWORD [rbp], 0x12345);
    enc!(mc => rx86::cmp_mi(&mut mc, (13, 0), 0x12345), cmp QWORD [r13], 0x12345);
    enc!(mc => rx86::cmp_mi(&mut mc, (15, 8), 0x12345), cmp QWORD [r15 + 8], 0x12345);
    // cmp_rm
    enc!(mc => rx86::cmp_rm(&mut mc, 0, (0, 0)), cmp rax, [rax]);
    enc!(mc => rx86::cmp_rm(&mut mc, 0, (0, 8)), cmp rax, [rax + 8]);
    enc!(mc => rx86::cmp_rm(&mut mc, 0, (0, -129)), cmp rax, [rax - 129]);
    enc!(mc => rx86::cmp_rm(&mut mc, 0, (0, 0x12345)), cmp rax, [rax + 0x12345]);
    enc!(mc => rx86::cmp_rm(&mut mc, 0, (4, 0)), cmp rax, [rsp]);
    enc!(mc => rx86::cmp_rm(&mut mc, 0, (4, 8)), cmp rax, [rsp + 8]);
    enc!(mc => rx86::cmp_rm(&mut mc, 0, (12, 0)), cmp rax, [r12]);
    enc!(mc => rx86::cmp_rm(&mut mc, 0, (12, 8)), cmp rax, [r12 + 8]);
    enc!(mc => rx86::cmp_rm(&mut mc, 0, (5, 0)), cmp rax, [rbp]);
    enc!(mc => rx86::cmp_rm(&mut mc, 0, (13, 0)), cmp rax, [r13]);
    enc!(mc => rx86::cmp_rm(&mut mc, 15, (15, 8)), cmp r15, [r15 + 8]);
    // cmp_mr
    enc!(mc => rx86::cmp_mr(&mut mc, (0, 0), 0), cmp QWORD [rax], rax);
    enc!(mc => rx86::cmp_mr(&mut mc, (0, 8), 0), cmp QWORD [rax + 8], rax);
    enc!(mc => rx86::cmp_mr(&mut mc, (0, -129), 0), cmp QWORD [rax - 129], rax);
    enc!(mc => rx86::cmp_mr(&mut mc, (0, 0x12345), 0), cmp QWORD [rax + 0x12345], rax);
    enc!(mc => rx86::cmp_mr(&mut mc, (4, 0), 0), cmp QWORD [rsp], rax);
    enc!(mc => rx86::cmp_mr(&mut mc, (4, 8), 0), cmp QWORD [rsp + 8], rax);
    enc!(mc => rx86::cmp_mr(&mut mc, (12, 0), 0), cmp QWORD [r12], rax);
    enc!(mc => rx86::cmp_mr(&mut mc, (12, 8), 0), cmp QWORD [r12 + 8], rax);
    enc!(mc => rx86::cmp_mr(&mut mc, (5, 0), 0), cmp QWORD [rbp], rax);
    enc!(mc => rx86::cmp_mr(&mut mc, (13, 0), 0), cmp QWORD [r13], rax);
    enc!(mc => rx86::cmp_mr(&mut mc, (15, 8), 15), cmp QWORD [r15 + 8], r15);
    // sub_rm
    enc!(mc => rx86::sub_rm(&mut mc, 0, (0, 0)), sub rax, [rax]);
    enc!(mc => rx86::sub_rm(&mut mc, 0, (0, 8)), sub rax, [rax + 8]);
    enc!(mc => rx86::sub_rm(&mut mc, 0, (0, -129)), sub rax, [rax - 129]);
    enc!(mc => rx86::sub_rm(&mut mc, 0, (0, 0x12345)), sub rax, [rax + 0x12345]);
    enc!(mc => rx86::sub_rm(&mut mc, 0, (4, 0)), sub rax, [rsp]);
    enc!(mc => rx86::sub_rm(&mut mc, 0, (4, 8)), sub rax, [rsp + 8]);
    enc!(mc => rx86::sub_rm(&mut mc, 0, (12, 0)), sub rax, [r12]);
    enc!(mc => rx86::sub_rm(&mut mc, 0, (12, 8)), sub rax, [r12 + 8]);
    enc!(mc => rx86::sub_rm(&mut mc, 0, (5, 0)), sub rax, [rbp]);
    enc!(mc => rx86::sub_rm(&mut mc, 0, (13, 0)), sub rax, [r13]);
    enc!(mc => rx86::sub_rm(&mut mc, 15, (15, 8)), sub r15, [r15 + 8]);
    // sub_mi
    enc!(mc => rx86::sub_mi(&mut mc, (0, 0), 0x12345), sub QWORD [rax], 0x12345);
    enc!(mc => rx86::sub_mi(&mut mc, (0, 8), 0x12345), sub QWORD [rax + 8], 0x12345);
    enc!(mc => rx86::sub_mi(&mut mc, (0, -129), 0x12345), sub QWORD [rax - 129], 0x12345);
    enc!(mc => rx86::sub_mi(&mut mc, (0, 0x12345), 0x12345), sub QWORD [rax + 0x12345], 0x12345);
    enc!(mc => rx86::sub_mi(&mut mc, (4, 0), 0x12345), sub QWORD [rsp], 0x12345);
    enc!(mc => rx86::sub_mi(&mut mc, (4, 8), 0x12345), sub QWORD [rsp + 8], 0x12345);
    enc!(mc => rx86::sub_mi(&mut mc, (12, 0), 0x12345), sub QWORD [r12], 0x12345);
    enc!(mc => rx86::sub_mi(&mut mc, (12, 8), 0x12345), sub QWORD [r12 + 8], 0x12345);
    enc!(mc => rx86::sub_mi(&mut mc, (5, 0), 0x12345), sub QWORD [rbp], 0x12345);
    enc!(mc => rx86::sub_mi(&mut mc, (13, 0), 0x12345), sub QWORD [r13], 0x12345);
    enc!(mc => rx86::sub_mi(&mut mc, (15, 8), 0x12345), sub QWORD [r15 + 8], 0x12345);
    // or8_mi
    enc!(mc => rx86::or8_mi(&mut mc, (0, 0), 0x11), or BYTE [rax], 0x11);
    enc!(mc => rx86::or8_mi(&mut mc, (0, 8), 0x11), or BYTE [rax + 8], 0x11);
    enc!(mc => rx86::or8_mi(&mut mc, (0, -129), 0x11), or BYTE [rax - 129], 0x11);
    enc!(mc => rx86::or8_mi(&mut mc, (0, 0x12345), 0x11), or BYTE [rax + 0x12345], 0x11);
    enc!(mc => rx86::or8_mi(&mut mc, (4, 0), 0x11), or BYTE [rsp], 0x11);
    enc!(mc => rx86::or8_mi(&mut mc, (4, 8), 0x11), or BYTE [rsp + 8], 0x11);
    enc!(mc => rx86::or8_mi(&mut mc, (12, 0), 0x11), or BYTE [r12], 0x11);
    enc!(mc => rx86::or8_mi(&mut mc, (12, 8), 0x11), or BYTE [r12 + 8], 0x11);
    enc!(mc => rx86::or8_mi(&mut mc, (5, 0), 0x11), or BYTE [rbp], 0x11);
    enc!(mc => rx86::or8_mi(&mut mc, (13, 0), 0x11), or BYTE [r13], 0x11);
    enc!(mc => rx86::or8_mi(&mut mc, (15, 8), 0x11), or BYTE [r15 + 8], 0x11);
    // lea_ra
    enc!(mc => rx86::lea_ra(&mut mc, 0, (0 as i16, 1, 0, 0)), lea rax, [rax + rcx * 1]);
    enc!(mc => rx86::lea_ra(&mut mc, 0, (0 as i16, 1, 0, 8)), lea rax, [rax + rcx * 1 + 8]);
    enc!(mc => rx86::lea_ra(&mut mc, 0, (0 as i16, 1, 3, 0)), lea rax, [rax + rcx * 8]);
    enc!(mc => rx86::lea_ra(&mut mc, 0, (0 as i16, 1, 3, 8)), lea rax, [rax + rcx * 8 + 8]);
    enc!(mc => rx86::lea_ra(&mut mc, 0, (0 as i16, 9, 0, 0)), lea rax, [rax + r9 * 1]);
    enc!(mc => rx86::lea_ra(&mut mc, 0, (0 as i16, 9, 0, 8)), lea rax, [rax + r9 * 1 + 8]);
    enc!(mc => rx86::lea_ra(&mut mc, 0, (0 as i16, 9, 3, 0)), lea rax, [rax + r9 * 8]);
    enc!(mc => rx86::lea_ra(&mut mc, 0, (0 as i16, 9, 3, 8)), lea rax, [rax + r9 * 8 + 8]);
    enc!(mc => rx86::lea_ra(&mut mc, 0, (13 as i16, 1, 0, 0)), lea rax, [r13 + rcx * 1]);
    enc!(mc => rx86::lea_ra(&mut mc, 0, (13 as i16, 1, 0, 8)), lea rax, [r13 + rcx * 1 + 8]);
    enc!(mc => rx86::lea_ra(&mut mc, 0, (13 as i16, 1, 3, 0)), lea rax, [r13 + rcx * 8]);
    enc!(mc => rx86::lea_ra(&mut mc, 0, (13 as i16, 1, 3, 8)), lea rax, [r13 + rcx * 8 + 8]);
    enc!(mc => rx86::lea_ra(&mut mc, 0, (13 as i16, 9, 0, 0)), lea rax, [r13 + r9 * 1]);
    enc!(mc => rx86::lea_ra(&mut mc, 0, (13 as i16, 9, 0, 8)), lea rax, [r13 + r9 * 1 + 8]);
    enc!(mc => rx86::lea_ra(&mut mc, 0, (13 as i16, 9, 3, 0)), lea rax, [r13 + r9 * 8]);
    enc!(mc => rx86::lea_ra(&mut mc, 15, (13 as i16, 9, 3, 8)), lea r15, [r13 + r9 * 8 + 8]);
    // mov_ra
    enc!(mc => rx86::mov_ra(&mut mc, 0, (0 as i16, 1, 0, 0)), mov rax, [rax + rcx * 1]);
    enc!(mc => rx86::mov_ra(&mut mc, 0, (0 as i16, 1, 0, 8)), mov rax, [rax + rcx * 1 + 8]);
    enc!(mc => rx86::mov_ra(&mut mc, 0, (0 as i16, 1, 3, 0)), mov rax, [rax + rcx * 8]);
    enc!(mc => rx86::mov_ra(&mut mc, 0, (0 as i16, 1, 3, 8)), mov rax, [rax + rcx * 8 + 8]);
    enc!(mc => rx86::mov_ra(&mut mc, 0, (0 as i16, 9, 0, 0)), mov rax, [rax + r9 * 1]);
    enc!(mc => rx86::mov_ra(&mut mc, 0, (0 as i16, 9, 0, 8)), mov rax, [rax + r9 * 1 + 8]);
    enc!(mc => rx86::mov_ra(&mut mc, 0, (0 as i16, 9, 3, 0)), mov rax, [rax + r9 * 8]);
    enc!(mc => rx86::mov_ra(&mut mc, 0, (0 as i16, 9, 3, 8)), mov rax, [rax + r9 * 8 + 8]);
    enc!(mc => rx86::mov_ra(&mut mc, 0, (13 as i16, 1, 0, 0)), mov rax, [r13 + rcx * 1]);
    enc!(mc => rx86::mov_ra(&mut mc, 0, (13 as i16, 1, 0, 8)), mov rax, [r13 + rcx * 1 + 8]);
    enc!(mc => rx86::mov_ra(&mut mc, 0, (13 as i16, 1, 3, 0)), mov rax, [r13 + rcx * 8]);
    enc!(mc => rx86::mov_ra(&mut mc, 0, (13 as i16, 1, 3, 8)), mov rax, [r13 + rcx * 8 + 8]);
    enc!(mc => rx86::mov_ra(&mut mc, 0, (13 as i16, 9, 0, 0)), mov rax, [r13 + r9 * 1]);
    enc!(mc => rx86::mov_ra(&mut mc, 0, (13 as i16, 9, 0, 8)), mov rax, [r13 + r9 * 1 + 8]);
    enc!(mc => rx86::mov_ra(&mut mc, 0, (13 as i16, 9, 3, 0)), mov rax, [r13 + r9 * 8]);
    enc!(mc => rx86::mov_ra(&mut mc, 15, (13 as i16, 9, 3, 8)), mov r15, [r13 + r9 * 8 + 8]);
    // mov_ar
    enc!(mc => rx86::mov_ar(&mut mc, (0 as i16, 1, 0, 0), 0), mov [rax + rcx * 1], rax);
    enc!(mc => rx86::mov_ar(&mut mc, (0 as i16, 1, 0, 8), 0), mov [rax + rcx * 1 + 8], rax);
    enc!(mc => rx86::mov_ar(&mut mc, (0 as i16, 1, 3, 0), 0), mov [rax + rcx * 8], rax);
    enc!(mc => rx86::mov_ar(&mut mc, (0 as i16, 1, 3, 8), 0), mov [rax + rcx * 8 + 8], rax);
    enc!(mc => rx86::mov_ar(&mut mc, (0 as i16, 9, 0, 0), 0), mov [rax + r9 * 1], rax);
    enc!(mc => rx86::mov_ar(&mut mc, (0 as i16, 9, 0, 8), 0), mov [rax + r9 * 1 + 8], rax);
    enc!(mc => rx86::mov_ar(&mut mc, (0 as i16, 9, 3, 0), 0), mov [rax + r9 * 8], rax);
    enc!(mc => rx86::mov_ar(&mut mc, (0 as i16, 9, 3, 8), 0), mov [rax + r9 * 8 + 8], rax);
    enc!(mc => rx86::mov_ar(&mut mc, (13 as i16, 1, 0, 0), 0), mov [r13 + rcx * 1], rax);
    enc!(mc => rx86::mov_ar(&mut mc, (13 as i16, 1, 0, 8), 0), mov [r13 + rcx * 1 + 8], rax);
    enc!(mc => rx86::mov_ar(&mut mc, (13 as i16, 1, 3, 0), 0), mov [r13 + rcx * 8], rax);
    enc!(mc => rx86::mov_ar(&mut mc, (13 as i16, 1, 3, 8), 0), mov [r13 + rcx * 8 + 8], rax);
    enc!(mc => rx86::mov_ar(&mut mc, (13 as i16, 9, 0, 0), 0), mov [r13 + r9 * 1], rax);
    enc!(mc => rx86::mov_ar(&mut mc, (13 as i16, 9, 0, 8), 0), mov [r13 + r9 * 1 + 8], rax);
    enc!(mc => rx86::mov_ar(&mut mc, (13 as i16, 9, 3, 0), 0), mov [r13 + r9 * 8], rax);
    enc!(mc => rx86::mov_ar(&mut mc, (13 as i16, 9, 3, 8), 15), mov [r13 + r9 * 8 + 8], r15);
    // mov32_ra
    enc!(mc => rx86::mov32_ra(&mut mc, 0, (0 as i16, 1, 0, 0)), mov eax, [rax + rcx * 1]);
    enc!(mc => rx86::mov32_ra(&mut mc, 0, (0 as i16, 1, 0, 8)), mov eax, [rax + rcx * 1 + 8]);
    enc!(mc => rx86::mov32_ra(&mut mc, 0, (0 as i16, 1, 3, 0)), mov eax, [rax + rcx * 8]);
    enc!(mc => rx86::mov32_ra(&mut mc, 0, (0 as i16, 1, 3, 8)), mov eax, [rax + rcx * 8 + 8]);
    enc!(mc => rx86::mov32_ra(&mut mc, 0, (0 as i16, 9, 0, 0)), mov eax, [rax + r9 * 1]);
    enc!(mc => rx86::mov32_ra(&mut mc, 0, (0 as i16, 9, 0, 8)), mov eax, [rax + r9 * 1 + 8]);
    enc!(mc => rx86::mov32_ra(&mut mc, 0, (0 as i16, 9, 3, 0)), mov eax, [rax + r9 * 8]);
    enc!(mc => rx86::mov32_ra(&mut mc, 0, (0 as i16, 9, 3, 8)), mov eax, [rax + r9 * 8 + 8]);
    enc!(mc => rx86::mov32_ra(&mut mc, 0, (13 as i16, 1, 0, 0)), mov eax, [r13 + rcx * 1]);
    enc!(mc => rx86::mov32_ra(&mut mc, 0, (13 as i16, 1, 0, 8)), mov eax, [r13 + rcx * 1 + 8]);
    enc!(mc => rx86::mov32_ra(&mut mc, 0, (13 as i16, 1, 3, 0)), mov eax, [r13 + rcx * 8]);
    enc!(mc => rx86::mov32_ra(&mut mc, 0, (13 as i16, 1, 3, 8)), mov eax, [r13 + rcx * 8 + 8]);
    enc!(mc => rx86::mov32_ra(&mut mc, 0, (13 as i16, 9, 0, 0)), mov eax, [r13 + r9 * 1]);
    enc!(mc => rx86::mov32_ra(&mut mc, 0, (13 as i16, 9, 0, 8)), mov eax, [r13 + r9 * 1 + 8]);
    enc!(mc => rx86::mov32_ra(&mut mc, 0, (13 as i16, 9, 3, 0)), mov eax, [r13 + r9 * 8]);
    enc!(mc => rx86::mov32_ra(&mut mc, 15, (13 as i16, 9, 3, 8)), mov r15d, [r13 + r9 * 8 + 8]);
    // mov32_ar
    enc!(mc => rx86::mov32_ar(&mut mc, (0 as i16, 1, 0, 0), 0), mov [rax + rcx * 1], eax);
    enc!(mc => rx86::mov32_ar(&mut mc, (0 as i16, 1, 0, 8), 0), mov [rax + rcx * 1 + 8], eax);
    enc!(mc => rx86::mov32_ar(&mut mc, (0 as i16, 1, 3, 0), 0), mov [rax + rcx * 8], eax);
    enc!(mc => rx86::mov32_ar(&mut mc, (0 as i16, 1, 3, 8), 0), mov [rax + rcx * 8 + 8], eax);
    enc!(mc => rx86::mov32_ar(&mut mc, (0 as i16, 9, 0, 0), 0), mov [rax + r9 * 1], eax);
    enc!(mc => rx86::mov32_ar(&mut mc, (0 as i16, 9, 0, 8), 0), mov [rax + r9 * 1 + 8], eax);
    enc!(mc => rx86::mov32_ar(&mut mc, (0 as i16, 9, 3, 0), 0), mov [rax + r9 * 8], eax);
    enc!(mc => rx86::mov32_ar(&mut mc, (0 as i16, 9, 3, 8), 0), mov [rax + r9 * 8 + 8], eax);
    enc!(mc => rx86::mov32_ar(&mut mc, (13 as i16, 1, 0, 0), 0), mov [r13 + rcx * 1], eax);
    enc!(mc => rx86::mov32_ar(&mut mc, (13 as i16, 1, 0, 8), 0), mov [r13 + rcx * 1 + 8], eax);
    enc!(mc => rx86::mov32_ar(&mut mc, (13 as i16, 1, 3, 0), 0), mov [r13 + rcx * 8], eax);
    enc!(mc => rx86::mov32_ar(&mut mc, (13 as i16, 1, 3, 8), 0), mov [r13 + rcx * 8 + 8], eax);
    enc!(mc => rx86::mov32_ar(&mut mc, (13 as i16, 9, 0, 0), 0), mov [r13 + r9 * 1], eax);
    enc!(mc => rx86::mov32_ar(&mut mc, (13 as i16, 9, 0, 8), 0), mov [r13 + r9 * 1 + 8], eax);
    enc!(mc => rx86::mov32_ar(&mut mc, (13 as i16, 9, 3, 0), 0), mov [r13 + r9 * 8], eax);
    enc!(mc => rx86::mov32_ar(&mut mc, (13 as i16, 9, 3, 8), 15), mov [r13 + r9 * 8 + 8], r15d);
    // mov16_ar
    enc!(mc => rx86::mov16_ar(&mut mc, (0 as i16, 1, 0, 0), 0), mov [rax + rcx * 1], ax);
    enc!(mc => rx86::mov16_ar(&mut mc, (0 as i16, 1, 0, 8), 0), mov [rax + rcx * 1 + 8], ax);
    enc!(mc => rx86::mov16_ar(&mut mc, (0 as i16, 1, 3, 0), 0), mov [rax + rcx * 8], ax);
    enc!(mc => rx86::mov16_ar(&mut mc, (0 as i16, 1, 3, 8), 0), mov [rax + rcx * 8 + 8], ax);
    enc!(mc => rx86::mov16_ar(&mut mc, (0 as i16, 9, 0, 0), 0), mov [rax + r9 * 1], ax);
    enc!(mc => rx86::mov16_ar(&mut mc, (0 as i16, 9, 0, 8), 0), mov [rax + r9 * 1 + 8], ax);
    enc!(mc => rx86::mov16_ar(&mut mc, (0 as i16, 9, 3, 0), 0), mov [rax + r9 * 8], ax);
    enc!(mc => rx86::mov16_ar(&mut mc, (0 as i16, 9, 3, 8), 0), mov [rax + r9 * 8 + 8], ax);
    enc!(mc => rx86::mov16_ar(&mut mc, (13 as i16, 1, 0, 0), 0), mov [r13 + rcx * 1], ax);
    enc!(mc => rx86::mov16_ar(&mut mc, (13 as i16, 1, 0, 8), 0), mov [r13 + rcx * 1 + 8], ax);
    enc!(mc => rx86::mov16_ar(&mut mc, (13 as i16, 1, 3, 0), 0), mov [r13 + rcx * 8], ax);
    enc!(mc => rx86::mov16_ar(&mut mc, (13 as i16, 1, 3, 8), 0), mov [r13 + rcx * 8 + 8], ax);
    enc!(mc => rx86::mov16_ar(&mut mc, (13 as i16, 9, 0, 0), 0), mov [r13 + r9 * 1], ax);
    enc!(mc => rx86::mov16_ar(&mut mc, (13 as i16, 9, 0, 8), 0), mov [r13 + r9 * 1 + 8], ax);
    enc!(mc => rx86::mov16_ar(&mut mc, (13 as i16, 9, 3, 0), 0), mov [r13 + r9 * 8], ax);
    enc!(mc => rx86::mov16_ar(&mut mc, (13 as i16, 9, 3, 8), 15), mov [r13 + r9 * 8 + 8], r15w);
    // mov8_ar
    enc!(mc => rx86::mov8_ar(&mut mc, (0 as i16, 1, 0, 0), 6), mov BYTE [rax + rcx * 1], sil);
    // `rex_fw` (`encode_rex`) always emits a REX byte, even `0x40`.
    enc_bytes!(mc => rx86::mov8_ar(&mut mc, (0 as i16, 1, 0, 8), 0), [0x40, 0x88, 0x44, 0x08, 0x08]);
    // `rex_fw` (`encode_rex`) always emits a REX byte, even `0x40`.
    enc_bytes!(mc => rx86::mov8_ar(&mut mc, (0 as i16, 1, 3, 0), 0), [0x40, 0x88, 0x04, 0xC8]);
    // `rex_fw` (`encode_rex`) always emits a REX byte, even `0x40`.
    enc_bytes!(mc => rx86::mov8_ar(&mut mc, (0 as i16, 1, 3, 8), 0), [0x40, 0x88, 0x44, 0xC8, 0x08]);
    enc!(mc => rx86::mov8_ar(&mut mc, (0 as i16, 9, 0, 0), 0), mov BYTE [rax + r9 * 1], al);
    enc!(mc => rx86::mov8_ar(&mut mc, (0 as i16, 9, 0, 8), 0), mov BYTE [rax + r9 * 1 + 8], al);
    enc!(mc => rx86::mov8_ar(&mut mc, (0 as i16, 9, 3, 0), 0), mov BYTE [rax + r9 * 8], al);
    enc!(mc => rx86::mov8_ar(&mut mc, (0 as i16, 9, 3, 8), 0), mov BYTE [rax + r9 * 8 + 8], al);
    enc!(mc => rx86::mov8_ar(&mut mc, (13 as i16, 1, 0, 0), 0), mov BYTE [r13 + rcx * 1], al);
    enc!(mc => rx86::mov8_ar(&mut mc, (13 as i16, 1, 0, 8), 0), mov BYTE [r13 + rcx * 1 + 8], al);
    enc!(mc => rx86::mov8_ar(&mut mc, (13 as i16, 1, 3, 0), 0), mov BYTE [r13 + rcx * 8], al);
    enc!(mc => rx86::mov8_ar(&mut mc, (13 as i16, 1, 3, 8), 0), mov BYTE [r13 + rcx * 8 + 8], al);
    enc!(mc => rx86::mov8_ar(&mut mc, (13 as i16, 9, 0, 0), 0), mov BYTE [r13 + r9 * 1], al);
    enc!(mc => rx86::mov8_ar(&mut mc, (13 as i16, 9, 0, 8), 0), mov BYTE [r13 + r9 * 1 + 8], al);
    enc!(mc => rx86::mov8_ar(&mut mc, (13 as i16, 9, 3, 0), 0), mov BYTE [r13 + r9 * 8], al);
    enc!(mc => rx86::mov8_ar(&mut mc, (13 as i16, 9, 3, 8), 8), mov BYTE [r13 + r9 * 8 + 8], r8b);
    // movzx8_ra
    enc!(mc => rx86::movzx8_ra(&mut mc, 0, (0 as i16, 1, 0, 0)), movzx rax, BYTE [rax + rcx * 1]);
    enc!(mc => rx86::movzx8_ra(&mut mc, 0, (0 as i16, 1, 0, 8)), movzx rax, BYTE [rax + rcx * 1 + 8]);
    enc!(mc => rx86::movzx8_ra(&mut mc, 0, (0 as i16, 1, 3, 0)), movzx rax, BYTE [rax + rcx * 8]);
    enc!(mc => rx86::movzx8_ra(&mut mc, 0, (0 as i16, 1, 3, 8)), movzx rax, BYTE [rax + rcx * 8 + 8]);
    enc!(mc => rx86::movzx8_ra(&mut mc, 0, (0 as i16, 9, 0, 0)), movzx rax, BYTE [rax + r9 * 1]);
    enc!(mc => rx86::movzx8_ra(&mut mc, 0, (0 as i16, 9, 0, 8)), movzx rax, BYTE [rax + r9 * 1 + 8]);
    enc!(mc => rx86::movzx8_ra(&mut mc, 0, (0 as i16, 9, 3, 0)), movzx rax, BYTE [rax + r9 * 8]);
    enc!(mc => rx86::movzx8_ra(&mut mc, 0, (0 as i16, 9, 3, 8)), movzx rax, BYTE [rax + r9 * 8 + 8]);
    enc!(mc => rx86::movzx8_ra(&mut mc, 0, (13 as i16, 1, 0, 0)), movzx rax, BYTE [r13 + rcx * 1]);
    enc!(mc => rx86::movzx8_ra(&mut mc, 0, (13 as i16, 1, 0, 8)), movzx rax, BYTE [r13 + rcx * 1 + 8]);
    enc!(mc => rx86::movzx8_ra(&mut mc, 0, (13 as i16, 1, 3, 0)), movzx rax, BYTE [r13 + rcx * 8]);
    enc!(mc => rx86::movzx8_ra(&mut mc, 0, (13 as i16, 1, 3, 8)), movzx rax, BYTE [r13 + rcx * 8 + 8]);
    enc!(mc => rx86::movzx8_ra(&mut mc, 0, (13 as i16, 9, 0, 0)), movzx rax, BYTE [r13 + r9 * 1]);
    enc!(mc => rx86::movzx8_ra(&mut mc, 0, (13 as i16, 9, 0, 8)), movzx rax, BYTE [r13 + r9 * 1 + 8]);
    enc!(mc => rx86::movzx8_ra(&mut mc, 0, (13 as i16, 9, 3, 0)), movzx rax, BYTE [r13 + r9 * 8]);
    enc!(mc => rx86::movzx8_ra(&mut mc, 15, (13 as i16, 9, 3, 8)), movzx r15, BYTE [r13 + r9 * 8 + 8]);
    // movzx16_ra
    enc!(mc => rx86::movzx16_ra(&mut mc, 0, (0 as i16, 1, 0, 0)), movzx rax, WORD [rax + rcx * 1]);
    enc!(mc => rx86::movzx16_ra(&mut mc, 0, (0 as i16, 1, 0, 8)), movzx rax, WORD [rax + rcx * 1 + 8]);
    enc!(mc => rx86::movzx16_ra(&mut mc, 0, (0 as i16, 1, 3, 0)), movzx rax, WORD [rax + rcx * 8]);
    enc!(mc => rx86::movzx16_ra(&mut mc, 0, (0 as i16, 1, 3, 8)), movzx rax, WORD [rax + rcx * 8 + 8]);
    enc!(mc => rx86::movzx16_ra(&mut mc, 0, (0 as i16, 9, 0, 0)), movzx rax, WORD [rax + r9 * 1]);
    enc!(mc => rx86::movzx16_ra(&mut mc, 0, (0 as i16, 9, 0, 8)), movzx rax, WORD [rax + r9 * 1 + 8]);
    enc!(mc => rx86::movzx16_ra(&mut mc, 0, (0 as i16, 9, 3, 0)), movzx rax, WORD [rax + r9 * 8]);
    enc!(mc => rx86::movzx16_ra(&mut mc, 0, (0 as i16, 9, 3, 8)), movzx rax, WORD [rax + r9 * 8 + 8]);
    enc!(mc => rx86::movzx16_ra(&mut mc, 0, (13 as i16, 1, 0, 0)), movzx rax, WORD [r13 + rcx * 1]);
    enc!(mc => rx86::movzx16_ra(&mut mc, 0, (13 as i16, 1, 0, 8)), movzx rax, WORD [r13 + rcx * 1 + 8]);
    enc!(mc => rx86::movzx16_ra(&mut mc, 0, (13 as i16, 1, 3, 0)), movzx rax, WORD [r13 + rcx * 8]);
    enc!(mc => rx86::movzx16_ra(&mut mc, 0, (13 as i16, 1, 3, 8)), movzx rax, WORD [r13 + rcx * 8 + 8]);
    enc!(mc => rx86::movzx16_ra(&mut mc, 0, (13 as i16, 9, 0, 0)), movzx rax, WORD [r13 + r9 * 1]);
    enc!(mc => rx86::movzx16_ra(&mut mc, 0, (13 as i16, 9, 0, 8)), movzx rax, WORD [r13 + r9 * 1 + 8]);
    enc!(mc => rx86::movzx16_ra(&mut mc, 0, (13 as i16, 9, 3, 0)), movzx rax, WORD [r13 + r9 * 8]);
    enc!(mc => rx86::movzx16_ra(&mut mc, 15, (13 as i16, 9, 3, 8)), movzx r15, WORD [r13 + r9 * 8 + 8]);
    // movsx8_ra
    enc!(mc => rx86::movsx8_ra(&mut mc, 0, (0 as i16, 1, 0, 0)), movsx rax, BYTE [rax + rcx * 1]);
    enc!(mc => rx86::movsx8_ra(&mut mc, 0, (0 as i16, 1, 0, 8)), movsx rax, BYTE [rax + rcx * 1 + 8]);
    enc!(mc => rx86::movsx8_ra(&mut mc, 0, (0 as i16, 1, 3, 0)), movsx rax, BYTE [rax + rcx * 8]);
    enc!(mc => rx86::movsx8_ra(&mut mc, 0, (0 as i16, 1, 3, 8)), movsx rax, BYTE [rax + rcx * 8 + 8]);
    enc!(mc => rx86::movsx8_ra(&mut mc, 0, (0 as i16, 9, 0, 0)), movsx rax, BYTE [rax + r9 * 1]);
    enc!(mc => rx86::movsx8_ra(&mut mc, 0, (0 as i16, 9, 0, 8)), movsx rax, BYTE [rax + r9 * 1 + 8]);
    enc!(mc => rx86::movsx8_ra(&mut mc, 0, (0 as i16, 9, 3, 0)), movsx rax, BYTE [rax + r9 * 8]);
    enc!(mc => rx86::movsx8_ra(&mut mc, 0, (0 as i16, 9, 3, 8)), movsx rax, BYTE [rax + r9 * 8 + 8]);
    enc!(mc => rx86::movsx8_ra(&mut mc, 0, (13 as i16, 1, 0, 0)), movsx rax, BYTE [r13 + rcx * 1]);
    enc!(mc => rx86::movsx8_ra(&mut mc, 0, (13 as i16, 1, 0, 8)), movsx rax, BYTE [r13 + rcx * 1 + 8]);
    enc!(mc => rx86::movsx8_ra(&mut mc, 0, (13 as i16, 1, 3, 0)), movsx rax, BYTE [r13 + rcx * 8]);
    enc!(mc => rx86::movsx8_ra(&mut mc, 0, (13 as i16, 1, 3, 8)), movsx rax, BYTE [r13 + rcx * 8 + 8]);
    enc!(mc => rx86::movsx8_ra(&mut mc, 0, (13 as i16, 9, 0, 0)), movsx rax, BYTE [r13 + r9 * 1]);
    enc!(mc => rx86::movsx8_ra(&mut mc, 0, (13 as i16, 9, 0, 8)), movsx rax, BYTE [r13 + r9 * 1 + 8]);
    enc!(mc => rx86::movsx8_ra(&mut mc, 0, (13 as i16, 9, 3, 0)), movsx rax, BYTE [r13 + r9 * 8]);
    enc!(mc => rx86::movsx8_ra(&mut mc, 15, (13 as i16, 9, 3, 8)), movsx r15, BYTE [r13 + r9 * 8 + 8]);
    // movsx16_ra
    enc!(mc => rx86::movsx16_ra(&mut mc, 0, (0 as i16, 1, 0, 0)), movsx rax, WORD [rax + rcx * 1]);
    enc!(mc => rx86::movsx16_ra(&mut mc, 0, (0 as i16, 1, 0, 8)), movsx rax, WORD [rax + rcx * 1 + 8]);
    enc!(mc => rx86::movsx16_ra(&mut mc, 0, (0 as i16, 1, 3, 0)), movsx rax, WORD [rax + rcx * 8]);
    enc!(mc => rx86::movsx16_ra(&mut mc, 0, (0 as i16, 1, 3, 8)), movsx rax, WORD [rax + rcx * 8 + 8]);
    enc!(mc => rx86::movsx16_ra(&mut mc, 0, (0 as i16, 9, 0, 0)), movsx rax, WORD [rax + r9 * 1]);
    enc!(mc => rx86::movsx16_ra(&mut mc, 0, (0 as i16, 9, 0, 8)), movsx rax, WORD [rax + r9 * 1 + 8]);
    enc!(mc => rx86::movsx16_ra(&mut mc, 0, (0 as i16, 9, 3, 0)), movsx rax, WORD [rax + r9 * 8]);
    enc!(mc => rx86::movsx16_ra(&mut mc, 0, (0 as i16, 9, 3, 8)), movsx rax, WORD [rax + r9 * 8 + 8]);
    enc!(mc => rx86::movsx16_ra(&mut mc, 0, (13 as i16, 1, 0, 0)), movsx rax, WORD [r13 + rcx * 1]);
    enc!(mc => rx86::movsx16_ra(&mut mc, 0, (13 as i16, 1, 0, 8)), movsx rax, WORD [r13 + rcx * 1 + 8]);
    enc!(mc => rx86::movsx16_ra(&mut mc, 0, (13 as i16, 1, 3, 0)), movsx rax, WORD [r13 + rcx * 8]);
    enc!(mc => rx86::movsx16_ra(&mut mc, 0, (13 as i16, 1, 3, 8)), movsx rax, WORD [r13 + rcx * 8 + 8]);
    enc!(mc => rx86::movsx16_ra(&mut mc, 0, (13 as i16, 9, 0, 0)), movsx rax, WORD [r13 + r9 * 1]);
    enc!(mc => rx86::movsx16_ra(&mut mc, 0, (13 as i16, 9, 0, 8)), movsx rax, WORD [r13 + r9 * 1 + 8]);
    enc!(mc => rx86::movsx16_ra(&mut mc, 0, (13 as i16, 9, 3, 0)), movsx rax, WORD [r13 + r9 * 8]);
    enc!(mc => rx86::movsx16_ra(&mut mc, 15, (13 as i16, 9, 3, 8)), movsx r15, WORD [r13 + r9 * 8 + 8]);
    // movsx32_ra
    enc!(mc => rx86::movsx32_ra(&mut mc, 0, (0 as i16, 1, 0, 0)), movsxd rax, DWORD [rax + rcx * 1]);
    enc!(mc => rx86::movsx32_ra(&mut mc, 0, (0 as i16, 1, 0, 8)), movsxd rax, DWORD [rax + rcx * 1 + 8]);
    enc!(mc => rx86::movsx32_ra(&mut mc, 0, (0 as i16, 1, 3, 0)), movsxd rax, DWORD [rax + rcx * 8]);
    enc!(mc => rx86::movsx32_ra(&mut mc, 0, (0 as i16, 1, 3, 8)), movsxd rax, DWORD [rax + rcx * 8 + 8]);
    enc!(mc => rx86::movsx32_ra(&mut mc, 0, (0 as i16, 9, 0, 0)), movsxd rax, DWORD [rax + r9 * 1]);
    enc!(mc => rx86::movsx32_ra(&mut mc, 0, (0 as i16, 9, 0, 8)), movsxd rax, DWORD [rax + r9 * 1 + 8]);
    enc!(mc => rx86::movsx32_ra(&mut mc, 0, (0 as i16, 9, 3, 0)), movsxd rax, DWORD [rax + r9 * 8]);
    enc!(mc => rx86::movsx32_ra(&mut mc, 0, (0 as i16, 9, 3, 8)), movsxd rax, DWORD [rax + r9 * 8 + 8]);
    enc!(mc => rx86::movsx32_ra(&mut mc, 0, (13 as i16, 1, 0, 0)), movsxd rax, DWORD [r13 + rcx * 1]);
    enc!(mc => rx86::movsx32_ra(&mut mc, 0, (13 as i16, 1, 0, 8)), movsxd rax, DWORD [r13 + rcx * 1 + 8]);
    enc!(mc => rx86::movsx32_ra(&mut mc, 0, (13 as i16, 1, 3, 0)), movsxd rax, DWORD [r13 + rcx * 8]);
    enc!(mc => rx86::movsx32_ra(&mut mc, 0, (13 as i16, 1, 3, 8)), movsxd rax, DWORD [r13 + rcx * 8 + 8]);
    enc!(mc => rx86::movsx32_ra(&mut mc, 0, (13 as i16, 9, 0, 0)), movsxd rax, DWORD [r13 + r9 * 1]);
    enc!(mc => rx86::movsx32_ra(&mut mc, 0, (13 as i16, 9, 0, 8)), movsxd rax, DWORD [r13 + r9 * 1 + 8]);
    enc!(mc => rx86::movsx32_ra(&mut mc, 0, (13 as i16, 9, 3, 0)), movsxd rax, DWORD [r13 + r9 * 8]);
    enc!(mc => rx86::movsx32_ra(&mut mc, 15, (13 as i16, 9, 3, 8)), movsxd r15, DWORD [r13 + r9 * 8 + 8]);
    // movsd_xa
    enc!(mc => rx86::movsd_xa(&mut mc, 0, (0 as i16, 1, 0, 0)), movsd xmm0, [rax + rcx * 1]);
    enc!(mc => rx86::movsd_xa(&mut mc, 0, (0 as i16, 1, 0, 8)), movsd xmm0, [rax + rcx * 1 + 8]);
    enc!(mc => rx86::movsd_xa(&mut mc, 0, (0 as i16, 1, 3, 0)), movsd xmm0, [rax + rcx * 8]);
    enc!(mc => rx86::movsd_xa(&mut mc, 0, (0 as i16, 1, 3, 8)), movsd xmm0, [rax + rcx * 8 + 8]);
    enc!(mc => rx86::movsd_xa(&mut mc, 0, (0 as i16, 9, 0, 0)), movsd xmm0, [rax + r9 * 1]);
    enc!(mc => rx86::movsd_xa(&mut mc, 0, (0 as i16, 9, 0, 8)), movsd xmm0, [rax + r9 * 1 + 8]);
    enc!(mc => rx86::movsd_xa(&mut mc, 0, (0 as i16, 9, 3, 0)), movsd xmm0, [rax + r9 * 8]);
    enc!(mc => rx86::movsd_xa(&mut mc, 0, (0 as i16, 9, 3, 8)), movsd xmm0, [rax + r9 * 8 + 8]);
    enc!(mc => rx86::movsd_xa(&mut mc, 0, (13 as i16, 1, 0, 0)), movsd xmm0, [r13 + rcx * 1]);
    enc!(mc => rx86::movsd_xa(&mut mc, 0, (13 as i16, 1, 0, 8)), movsd xmm0, [r13 + rcx * 1 + 8]);
    enc!(mc => rx86::movsd_xa(&mut mc, 0, (13 as i16, 1, 3, 0)), movsd xmm0, [r13 + rcx * 8]);
    enc!(mc => rx86::movsd_xa(&mut mc, 0, (13 as i16, 1, 3, 8)), movsd xmm0, [r13 + rcx * 8 + 8]);
    enc!(mc => rx86::movsd_xa(&mut mc, 0, (13 as i16, 9, 0, 0)), movsd xmm0, [r13 + r9 * 1]);
    enc!(mc => rx86::movsd_xa(&mut mc, 0, (13 as i16, 9, 0, 8)), movsd xmm0, [r13 + r9 * 1 + 8]);
    enc!(mc => rx86::movsd_xa(&mut mc, 0, (13 as i16, 9, 3, 0)), movsd xmm0, [r13 + r9 * 8]);
    enc!(mc => rx86::movsd_xa(&mut mc, 15, (13 as i16, 9, 3, 8)), movsd xmm15, [r13 + r9 * 8 + 8]);
    // movsd_ax
    enc!(mc => rx86::movsd_ax(&mut mc, (0 as i16, 1, 0, 0), 0), movsd [rax + rcx * 1], xmm0);
    enc!(mc => rx86::movsd_ax(&mut mc, (0 as i16, 1, 0, 8), 0), movsd [rax + rcx * 1 + 8], xmm0);
    enc!(mc => rx86::movsd_ax(&mut mc, (0 as i16, 1, 3, 0), 0), movsd [rax + rcx * 8], xmm0);
    enc!(mc => rx86::movsd_ax(&mut mc, (0 as i16, 1, 3, 8), 0), movsd [rax + rcx * 8 + 8], xmm0);
    enc!(mc => rx86::movsd_ax(&mut mc, (0 as i16, 9, 0, 0), 0), movsd [rax + r9 * 1], xmm0);
    enc!(mc => rx86::movsd_ax(&mut mc, (0 as i16, 9, 0, 8), 0), movsd [rax + r9 * 1 + 8], xmm0);
    enc!(mc => rx86::movsd_ax(&mut mc, (0 as i16, 9, 3, 0), 0), movsd [rax + r9 * 8], xmm0);
    enc!(mc => rx86::movsd_ax(&mut mc, (0 as i16, 9, 3, 8), 0), movsd [rax + r9 * 8 + 8], xmm0);
    enc!(mc => rx86::movsd_ax(&mut mc, (13 as i16, 1, 0, 0), 0), movsd [r13 + rcx * 1], xmm0);
    enc!(mc => rx86::movsd_ax(&mut mc, (13 as i16, 1, 0, 8), 0), movsd [r13 + rcx * 1 + 8], xmm0);
    enc!(mc => rx86::movsd_ax(&mut mc, (13 as i16, 1, 3, 0), 0), movsd [r13 + rcx * 8], xmm0);
    enc!(mc => rx86::movsd_ax(&mut mc, (13 as i16, 1, 3, 8), 0), movsd [r13 + rcx * 8 + 8], xmm0);
    enc!(mc => rx86::movsd_ax(&mut mc, (13 as i16, 9, 0, 0), 0), movsd [r13 + r9 * 1], xmm0);
    enc!(mc => rx86::movsd_ax(&mut mc, (13 as i16, 9, 0, 8), 0), movsd [r13 + r9 * 1 + 8], xmm0);
    enc!(mc => rx86::movsd_ax(&mut mc, (13 as i16, 9, 3, 0), 0), movsd [r13 + r9 * 8], xmm0);
    enc!(mc => rx86::movsd_ax(&mut mc, (13 as i16, 9, 3, 8), 15), movsd [r13 + r9 * 8 + 8], xmm15);
    // movss_xa
    enc!(mc => rx86::movss_xa(&mut mc, 0, (0 as i16, 1, 0, 0)), movss xmm0, [rax + rcx * 1]);
    enc!(mc => rx86::movss_xa(&mut mc, 0, (0 as i16, 1, 0, 8)), movss xmm0, [rax + rcx * 1 + 8]);
    enc!(mc => rx86::movss_xa(&mut mc, 0, (0 as i16, 1, 3, 0)), movss xmm0, [rax + rcx * 8]);
    enc!(mc => rx86::movss_xa(&mut mc, 0, (0 as i16, 1, 3, 8)), movss xmm0, [rax + rcx * 8 + 8]);
    enc!(mc => rx86::movss_xa(&mut mc, 0, (0 as i16, 9, 0, 0)), movss xmm0, [rax + r9 * 1]);
    enc!(mc => rx86::movss_xa(&mut mc, 0, (0 as i16, 9, 0, 8)), movss xmm0, [rax + r9 * 1 + 8]);
    enc!(mc => rx86::movss_xa(&mut mc, 0, (0 as i16, 9, 3, 0)), movss xmm0, [rax + r9 * 8]);
    enc!(mc => rx86::movss_xa(&mut mc, 0, (0 as i16, 9, 3, 8)), movss xmm0, [rax + r9 * 8 + 8]);
    enc!(mc => rx86::movss_xa(&mut mc, 0, (13 as i16, 1, 0, 0)), movss xmm0, [r13 + rcx * 1]);
    enc!(mc => rx86::movss_xa(&mut mc, 0, (13 as i16, 1, 0, 8)), movss xmm0, [r13 + rcx * 1 + 8]);
    enc!(mc => rx86::movss_xa(&mut mc, 0, (13 as i16, 1, 3, 0)), movss xmm0, [r13 + rcx * 8]);
    enc!(mc => rx86::movss_xa(&mut mc, 0, (13 as i16, 1, 3, 8)), movss xmm0, [r13 + rcx * 8 + 8]);
    enc!(mc => rx86::movss_xa(&mut mc, 0, (13 as i16, 9, 0, 0)), movss xmm0, [r13 + r9 * 1]);
    enc!(mc => rx86::movss_xa(&mut mc, 0, (13 as i16, 9, 0, 8)), movss xmm0, [r13 + r9 * 1 + 8]);
    enc!(mc => rx86::movss_xa(&mut mc, 0, (13 as i16, 9, 3, 0)), movss xmm0, [r13 + r9 * 8]);
    enc!(mc => rx86::movss_xa(&mut mc, 15, (13 as i16, 9, 3, 8)), movss xmm15, [r13 + r9 * 8 + 8]);
    // movss_ax
    enc!(mc => rx86::movss_ax(&mut mc, (0 as i16, 1, 0, 0), 0), movss [rax + rcx * 1], xmm0);
    enc!(mc => rx86::movss_ax(&mut mc, (0 as i16, 1, 0, 8), 0), movss [rax + rcx * 1 + 8], xmm0);
    enc!(mc => rx86::movss_ax(&mut mc, (0 as i16, 1, 3, 0), 0), movss [rax + rcx * 8], xmm0);
    enc!(mc => rx86::movss_ax(&mut mc, (0 as i16, 1, 3, 8), 0), movss [rax + rcx * 8 + 8], xmm0);
    enc!(mc => rx86::movss_ax(&mut mc, (0 as i16, 9, 0, 0), 0), movss [rax + r9 * 1], xmm0);
    enc!(mc => rx86::movss_ax(&mut mc, (0 as i16, 9, 0, 8), 0), movss [rax + r9 * 1 + 8], xmm0);
    enc!(mc => rx86::movss_ax(&mut mc, (0 as i16, 9, 3, 0), 0), movss [rax + r9 * 8], xmm0);
    enc!(mc => rx86::movss_ax(&mut mc, (0 as i16, 9, 3, 8), 0), movss [rax + r9 * 8 + 8], xmm0);
    enc!(mc => rx86::movss_ax(&mut mc, (13 as i16, 1, 0, 0), 0), movss [r13 + rcx * 1], xmm0);
    enc!(mc => rx86::movss_ax(&mut mc, (13 as i16, 1, 0, 8), 0), movss [r13 + rcx * 1 + 8], xmm0);
    enc!(mc => rx86::movss_ax(&mut mc, (13 as i16, 1, 3, 0), 0), movss [r13 + rcx * 8], xmm0);
    enc!(mc => rx86::movss_ax(&mut mc, (13 as i16, 1, 3, 8), 0), movss [r13 + rcx * 8 + 8], xmm0);
    enc!(mc => rx86::movss_ax(&mut mc, (13 as i16, 9, 0, 0), 0), movss [r13 + r9 * 1], xmm0);
    enc!(mc => rx86::movss_ax(&mut mc, (13 as i16, 9, 0, 8), 0), movss [r13 + r9 * 1 + 8], xmm0);
    enc!(mc => rx86::movss_ax(&mut mc, (13 as i16, 9, 3, 0), 0), movss [r13 + r9 * 8], xmm0);
    enc!(mc => rx86::movss_ax(&mut mc, (13 as i16, 9, 3, 8), 15), movss [r13 + r9 * 8 + 8], xmm15);
    // mov_ai
    enc!(mc => rx86::mov_ai(&mut mc, (0 as i16, 1, 0, 0), -2), mov QWORD [rax + rcx * 1], -2);
    enc!(mc => rx86::mov_ai(&mut mc, (0 as i16, 1, 0, 8), -2), mov QWORD [rax + rcx * 1 + 8], -2);
    enc!(mc => rx86::mov_ai(&mut mc, (0 as i16, 1, 3, 0), -2), mov QWORD [rax + rcx * 8], -2);
    enc!(mc => rx86::mov_ai(&mut mc, (0 as i16, 1, 3, 8), -2), mov QWORD [rax + rcx * 8 + 8], -2);
    enc!(mc => rx86::mov_ai(&mut mc, (0 as i16, 9, 0, 0), -2), mov QWORD [rax + r9 * 1], -2);
    enc!(mc => rx86::mov_ai(&mut mc, (0 as i16, 9, 0, 8), -2), mov QWORD [rax + r9 * 1 + 8], -2);
    enc!(mc => rx86::mov_ai(&mut mc, (0 as i16, 9, 3, 0), -2), mov QWORD [rax + r9 * 8], -2);
    enc!(mc => rx86::mov_ai(&mut mc, (0 as i16, 9, 3, 8), -2), mov QWORD [rax + r9 * 8 + 8], -2);
    enc!(mc => rx86::mov_ai(&mut mc, (13 as i16, 1, 0, 0), -2), mov QWORD [r13 + rcx * 1], -2);
    enc!(mc => rx86::mov_ai(&mut mc, (13 as i16, 1, 0, 8), -2), mov QWORD [r13 + rcx * 1 + 8], -2);
    enc!(mc => rx86::mov_ai(&mut mc, (13 as i16, 1, 3, 0), -2), mov QWORD [r13 + rcx * 8], -2);
    enc!(mc => rx86::mov_ai(&mut mc, (13 as i16, 1, 3, 8), -2), mov QWORD [r13 + rcx * 8 + 8], -2);
    enc!(mc => rx86::mov_ai(&mut mc, (13 as i16, 9, 0, 0), -2), mov QWORD [r13 + r9 * 1], -2);
    enc!(mc => rx86::mov_ai(&mut mc, (13 as i16, 9, 0, 8), -2), mov QWORD [r13 + r9 * 1 + 8], -2);
    enc!(mc => rx86::mov_ai(&mut mc, (13 as i16, 9, 3, 0), -2), mov QWORD [r13 + r9 * 8], -2);
    enc!(mc => rx86::mov_ai(&mut mc, (13 as i16, 9, 3, 8), -2), mov QWORD [r13 + r9 * 8 + 8], -2);
    // mov32_ai
    enc!(mc => rx86::mov32_ai(&mut mc, (0 as i16, 1, 0, 0), 0x123456), mov DWORD [rax + rcx * 1], 0x123456);
    enc!(mc => rx86::mov32_ai(&mut mc, (0 as i16, 1, 0, 8), 0x123456), mov DWORD [rax + rcx * 1 + 8], 0x123456);
    enc!(mc => rx86::mov32_ai(&mut mc, (0 as i16, 1, 3, 0), 0x123456), mov DWORD [rax + rcx * 8], 0x123456);
    enc!(mc => rx86::mov32_ai(&mut mc, (0 as i16, 1, 3, 8), 0x123456), mov DWORD [rax + rcx * 8 + 8], 0x123456);
    enc!(mc => rx86::mov32_ai(&mut mc, (0 as i16, 9, 0, 0), 0x123456), mov DWORD [rax + r9 * 1], 0x123456);
    enc!(mc => rx86::mov32_ai(&mut mc, (0 as i16, 9, 0, 8), 0x123456), mov DWORD [rax + r9 * 1 + 8], 0x123456);
    enc!(mc => rx86::mov32_ai(&mut mc, (0 as i16, 9, 3, 0), 0x123456), mov DWORD [rax + r9 * 8], 0x123456);
    enc!(mc => rx86::mov32_ai(&mut mc, (0 as i16, 9, 3, 8), 0x123456), mov DWORD [rax + r9 * 8 + 8], 0x123456);
    enc!(mc => rx86::mov32_ai(&mut mc, (13 as i16, 1, 0, 0), 0x123456), mov DWORD [r13 + rcx * 1], 0x123456);
    enc!(mc => rx86::mov32_ai(&mut mc, (13 as i16, 1, 0, 8), 0x123456), mov DWORD [r13 + rcx * 1 + 8], 0x123456);
    enc!(mc => rx86::mov32_ai(&mut mc, (13 as i16, 1, 3, 0), 0x123456), mov DWORD [r13 + rcx * 8], 0x123456);
    enc!(mc => rx86::mov32_ai(&mut mc, (13 as i16, 1, 3, 8), 0x123456), mov DWORD [r13 + rcx * 8 + 8], 0x123456);
    enc!(mc => rx86::mov32_ai(&mut mc, (13 as i16, 9, 0, 0), 0x123456), mov DWORD [r13 + r9 * 1], 0x123456);
    enc!(mc => rx86::mov32_ai(&mut mc, (13 as i16, 9, 0, 8), 0x123456), mov DWORD [r13 + r9 * 1 + 8], 0x123456);
    enc!(mc => rx86::mov32_ai(&mut mc, (13 as i16, 9, 3, 0), 0x123456), mov DWORD [r13 + r9 * 8], 0x123456);
    enc!(mc => rx86::mov32_ai(&mut mc, (13 as i16, 9, 3, 8), 0x123456), mov DWORD [r13 + r9 * 8 + 8], 0x123456);
    // mov16_ai
    enc!(mc => rx86::mov16_ai(&mut mc, (0 as i16, 1, 0, 0), 0x1234), mov WORD [rax + rcx * 1], 0x1234);
    enc!(mc => rx86::mov16_ai(&mut mc, (0 as i16, 1, 0, 8), 0x1234), mov WORD [rax + rcx * 1 + 8], 0x1234);
    enc!(mc => rx86::mov16_ai(&mut mc, (0 as i16, 1, 3, 0), 0x1234), mov WORD [rax + rcx * 8], 0x1234);
    enc!(mc => rx86::mov16_ai(&mut mc, (0 as i16, 1, 3, 8), 0x1234), mov WORD [rax + rcx * 8 + 8], 0x1234);
    enc!(mc => rx86::mov16_ai(&mut mc, (0 as i16, 9, 0, 0), 0x1234), mov WORD [rax + r9 * 1], 0x1234);
    enc!(mc => rx86::mov16_ai(&mut mc, (0 as i16, 9, 0, 8), 0x1234), mov WORD [rax + r9 * 1 + 8], 0x1234);
    enc!(mc => rx86::mov16_ai(&mut mc, (0 as i16, 9, 3, 0), 0x1234), mov WORD [rax + r9 * 8], 0x1234);
    enc!(mc => rx86::mov16_ai(&mut mc, (0 as i16, 9, 3, 8), 0x1234), mov WORD [rax + r9 * 8 + 8], 0x1234);
    enc!(mc => rx86::mov16_ai(&mut mc, (13 as i16, 1, 0, 0), 0x1234), mov WORD [r13 + rcx * 1], 0x1234);
    enc!(mc => rx86::mov16_ai(&mut mc, (13 as i16, 1, 0, 8), 0x1234), mov WORD [r13 + rcx * 1 + 8], 0x1234);
    enc!(mc => rx86::mov16_ai(&mut mc, (13 as i16, 1, 3, 0), 0x1234), mov WORD [r13 + rcx * 8], 0x1234);
    enc!(mc => rx86::mov16_ai(&mut mc, (13 as i16, 1, 3, 8), 0x1234), mov WORD [r13 + rcx * 8 + 8], 0x1234);
    enc!(mc => rx86::mov16_ai(&mut mc, (13 as i16, 9, 0, 0), 0x1234), mov WORD [r13 + r9 * 1], 0x1234);
    enc!(mc => rx86::mov16_ai(&mut mc, (13 as i16, 9, 0, 8), 0x1234), mov WORD [r13 + r9 * 1 + 8], 0x1234);
    enc!(mc => rx86::mov16_ai(&mut mc, (13 as i16, 9, 3, 0), 0x1234), mov WORD [r13 + r9 * 8], 0x1234);
    enc!(mc => rx86::mov16_ai(&mut mc, (13 as i16, 9, 3, 8), 0x1234), mov WORD [r13 + r9 * 8 + 8], 0x1234);
    // mov8_ai
    // `rex_fw` (`encode_rex`) always emits a REX byte, even `0x40`.
    enc_bytes!(mc => rx86::mov8_ai(&mut mc, (0 as i16, 1, 0, 0), -5), [0x40, 0xC6, 0x04, 0x08, 0xFB]);
    // `rex_fw` (`encode_rex`) always emits a REX byte, even `0x40`.
    enc_bytes!(mc => rx86::mov8_ai(&mut mc, (0 as i16, 1, 0, 8), -5), [0x40, 0xC6, 0x44, 0x08, 0x08, 0xFB]);
    // `rex_fw` (`encode_rex`) always emits a REX byte, even `0x40`.
    enc_bytes!(mc => rx86::mov8_ai(&mut mc, (0 as i16, 1, 3, 0), -5), [0x40, 0xC6, 0x04, 0xC8, 0xFB]);
    // `rex_fw` (`encode_rex`) always emits a REX byte, even `0x40`.
    enc_bytes!(mc => rx86::mov8_ai(&mut mc, (0 as i16, 1, 3, 8), -5), [0x40, 0xC6, 0x44, 0xC8, 0x08, 0xFB]);
    enc!(mc => rx86::mov8_ai(&mut mc, (0 as i16, 9, 0, 0), -5), mov BYTE [rax + r9 * 1], -5);
    enc!(mc => rx86::mov8_ai(&mut mc, (0 as i16, 9, 0, 8), -5), mov BYTE [rax + r9 * 1 + 8], -5);
    enc!(mc => rx86::mov8_ai(&mut mc, (0 as i16, 9, 3, 0), -5), mov BYTE [rax + r9 * 8], -5);
    enc!(mc => rx86::mov8_ai(&mut mc, (0 as i16, 9, 3, 8), -5), mov BYTE [rax + r9 * 8 + 8], -5);
    enc!(mc => rx86::mov8_ai(&mut mc, (13 as i16, 1, 0, 0), -5), mov BYTE [r13 + rcx * 1], -5);
    enc!(mc => rx86::mov8_ai(&mut mc, (13 as i16, 1, 0, 8), -5), mov BYTE [r13 + rcx * 1 + 8], -5);
    enc!(mc => rx86::mov8_ai(&mut mc, (13 as i16, 1, 3, 0), -5), mov BYTE [r13 + rcx * 8], -5);
    enc!(mc => rx86::mov8_ai(&mut mc, (13 as i16, 1, 3, 8), -5), mov BYTE [r13 + rcx * 8 + 8], -5);
    enc!(mc => rx86::mov8_ai(&mut mc, (13 as i16, 9, 0, 0), -5), mov BYTE [r13 + r9 * 1], -5);
    enc!(mc => rx86::mov8_ai(&mut mc, (13 as i16, 9, 0, 8), -5), mov BYTE [r13 + r9 * 1 + 8], -5);
    enc!(mc => rx86::mov8_ai(&mut mc, (13 as i16, 9, 3, 0), -5), mov BYTE [r13 + r9 * 8], -5);
    enc!(mc => rx86::mov8_ai(&mut mc, (13 as i16, 9, 3, 8), -5), mov BYTE [r13 + r9 * 8 + 8], -5);
}

#[test]
fn single_byte_and_fits_in_32bits() {
    assert!(rx86::single_byte(0) && rx86::single_byte(-128) && !rx86::single_byte(128));
    assert!(rx86::fits_in_32bits(0x12345) && !rx86::fits_in_32bits(2147483648));
}

#[test]
fn call_mem_encoding() {
    // `CALL_s` / `CALL_b`: `FF /2`, no REX.W.
    enc_bytes!(mc => rx86::call_s(&mut mc, 0), [0xFF, 0x14, 0x24]);
    enc_bytes!(mc => rx86::call_s(&mut mc, 32), [0xFF, 0x54, 0x24, 0x20]);
    enc_bytes!(mc => rx86::call_s(&mut mc, 128), [0xFF, 0x94, 0x24, 0x80, 0x00, 0x00, 0x00]);
    enc_bytes!(mc => rx86::call_b(&mut mc, 8), [0xFF, 0x55, 0x08]);
    enc_bytes!(mc => rx86::call_b(&mut mc, 128), [0xFF, 0x95, 0x80, 0x00, 0x00, 0x00]);
}
